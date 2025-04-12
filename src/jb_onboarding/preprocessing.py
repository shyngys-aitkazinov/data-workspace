import base64
import json
import os
import re
import zipfile
from datetime import datetime
from io import BytesIO
from os import PathLike

import fitz
import matplotlib.pyplot as plt
import torch
from docx import Document
from PIL import Image
from PyPDF2 import PdfReader
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer

from jb_onboarding.constants import DOCS


class PassportParser:
    def __init__(self, model_id: str = "OpenGVLab/InternVL2_5-1B"):
        """
        Initializes the PassportParser instance by loading the specified model
        and its corresponding tokenizer. This instance is intended for OCR
        tasks on passport images using the InternVL2_5 series.

        Args:
            model_id (str): Hugging Face model identifier for the InternVL2_5 variant.
                            Defaults to "OpenGVLab/InternVL2_5-1B".
        """
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float32,  # Use full precision on CPU
            low_cpu_mem_usage=False,
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    def __call__(self, path_to_passport: PathLike) -> str:
        """
        Processes the passport image by loading and preprocessing it,
        then runs OCR inference to transcribe all the text visible in the image.

        Args:
            path_to_passport (PathLike): File path to the passport image.

        Returns:
            str: The transcribed text extracted from the passport image.
        """
        # Open the image and convert it to RGB.
        image = Image.open(path_to_passport).convert("RGB")

        # Define the transformations: resize to 448x448, convert to tensor, and normalize.
        transform = transforms.Compose(
            [
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        # Apply the transformations and add a batch dimension.
        image_tensor = transform(image)
        pixel_values = image_tensor.unsqueeze(0)  # Shape: [1, 3, 448, 448]

        # Prepare the OCR prompt using the special <image> token.
        prompt = (
            "<image>\n"
            "Please extract the text from the passport image and output the information in a valid JSON object with exactly the following keys:\n"
            "  - Surname\n"
            "  - Code\n"
            "  - Passport Number\n"
            "  - Given Name\n"
            "  - Birth Date\n"
            "  - Citizenship\n"
            "  - Sex\n"
            "  - Issue Date\n"
            "  - Expiry Date\n"
            "  - MRZ line 1\n"
            "  - MRZ line 2\n"
            "For any field that is missing in the image, use an empty string as the value. Do not include any extra keys or text outside of the JSON."
        )

        # Use the model's chat method to generate the OCR transcription.
        output = self.model.chat(
            self.tokenizer, pixel_values, prompt, generation_config={"max_new_tokens": 512, "do_sample": False}
        )

        # Decode the output and convert it to a JSON object.
        passport_data = json.loads(output[8:-3])

        # for key in passport_data:
        #     if "Date" in key:
        #         # Convert date strings to the format
        #         date_str = passport_data[key]
        #         if date_str:
        #             try:
        #                 passport_data[key] = datetime.strptime(date_str, "%d-%b-%Y")
        #             except ValueError:
        #                 print(f"Invalid date format for {key}: {date_str}")

        # get the signature
        signature = image.crop((239, 210, 359, 242)).tobytes()
        image.crop((239, 213, 359, 242)).save("signature.png")
        passport_data["Signature"] = base64.b64encode(signature).decode("utf-8")

        return passport_data


class Preprocessor:
    def __init__(self, *args, **kwargs):
        """
        Initializes the Preprocessor instance. This class is responsible for
        extracting and processing data from a zip file containing various
        documents related to a user's profile.

        Args:
            **args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.args = args
        self.kwargs = kwargs
        self.passport_parser = PassportParser()

    def __call__(self, path_to_zip):
        temp_dir = str(path_to_zip).replace(".zip", "")
        with zipfile.ZipFile(path_to_zip, "r") as zip_ref:
            zip_ref.extractall(temp_dir)
            names = zip_ref.namelist()
            # profile.docx
            if "profile.docx" not in names:
                raise FileNotFoundError("profile.docx not found in the zip archive.")
            with zip_ref.open("profile.docx") as profile_file:
                doc = Document(profile_file)
                profile_json = self._parse_profile_docx_to_json(doc)

            # passport.png
            if "passport.png" not in names:
                raise FileNotFoundError("passport.png not found in the zip archive.")
            passport_data = self.passport_parser(os.path.join(temp_dir, "passport.png"))

            # description.txt
            if "description.txt" not in names:
                raise FileNotFoundError("description.txt not found in the zip archive.")
            with zip_ref.open("description.txt") as desc_file:
                raw_desc = desc_file.read().decode("utf-8")
                description_json = self._parse_description(raw_desc)

            # account.pdf
            if "account.pdf" not in names:
                raise FileNotFoundError("account.pdf not found in the zip archive.")
            with zip_ref.open("account.pdf") as pdf_file:
                account_pdf_bytes = pdf_file.read()
        account_parsed = self._extract_form_data_and_signature(account_pdf_bytes)
        output = {
            "profile": profile_json,
            "passport": passport_data,
            "description": description_json,
            "account": account_parsed,
        }
        return output

    def _parse_tick_field(self, text):
        pattern = r"(☒|☐)\s*([^☒☐]+)"
        matches = re.findall(pattern, text)
        options = []
        selected = []
        for mark, option in matches:
            option = option.strip()
            options.append(option)
            if mark == "☒":
                selected.append(option)
        return {"options": options, "selected": selected}

    def _clean_text(self, text):
        return " ".join(text.replace("\t", " ").split())

    def _parse_profile_docx_to_json(self, doc):
        result = {}
        current_section = "Default"
        result[current_section] = {}
        for table in doc.tables:
            table_data = []
            for row in table.rows:
                row_values = [self._clean_text(cell.text) for cell in row.cells]
                if any(row_values):
                    table_data.append(row_values)
            if len(table_data) == 1 and len(table_data[0]) == 1:
                section_title = table_data[0][0]
                if section_title:
                    current_section = section_title
                    result[current_section] = {}
                continue
            for row in table_data:
                key = row[0] if row[0] else None
                # For the value, lookthrough the remaining cells for non-empty content.
                # (Sometimes there might be multiple pieces of information.)
                value_candidates = [cell for cell in row[1:] if cell]
                inline_value = None
                if not value_candidates and row[0]:
                    inline_kv_pattern = r"(\bE-?Mail\b|Telephone)\s+([\w@.\-+ ]+)"
                    inline_matches = re.findall(inline_kv_pattern, row[0])
                    for k, v in inline_matches:
                        key = k.strip()
                        inline_value = v.strip()
                        result[current_section][key] = inline_value
                    continue  # Skip further processing since we’ve already stored it
                if not value_candidates:
                    value = None
                elif len(value_candidates) == 1:
                    value = value_candidates[0]
                else:
                    value = value_candidates
                if value and isinstance(value, str) and ("☒" in value or "☐" in value):
                    value = self._parse_tick_field(value)
                elif isinstance(value, list):
                    new_vals = []
                    for v in value:
                        if "☒" in v or "☐" in v:
                            new_vals.append(self._parse_tick_field(v))
                        else:
                            new_vals.append(v)
                    value = new_vals
                if key in result[current_section]:
                    if not isinstance(result[current_section][key], list):
                        result[current_section][key] = [result[current_section][key]]
                    result[current_section][key].append(value)
                else:
                    result[current_section][key] = value
        return result

    def _parse_description(self, text: str) -> dict:
        lines = text.splitlines()
        result = {}
        current_key = None
        current_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current_key is not None and current_lines and current_lines[-1] != "":
                    current_lines.append("")
                continue
            if stripped.endswith(":"):
                if current_key is not None:
                    result[current_key] = "\n".join(current_lines).strip()
                current_key = stripped[:-1].strip()
                current_lines = []
            else:
                if current_key is not None:
                    current_lines.append(stripped)
        if current_key is not None:
            result[current_key] = "\n".join(current_lines).strip()
        return result

    def _extract_form_fields(self, pdf_bytes):
        pdf_file = BytesIO(pdf_bytes)
        reader = PdfReader(pdf_file)
        fields = reader.get_fields()
        form_data = {}
        if fields:
            for field_name, field_info in fields.items():
                value = field_info.get("/V", "")
                form_data[field_name] = value
        else:
            print("No AcroForm fields found or the PDF is not a standard fillable form.")
        return form_data

    def _extract_signature_by_coordinates(self, pdf_bytes, ref_rect=(71, 582, 248, 625), dpi=300):
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if doc.page_count < 1:
            print("PDF has no pages.")
            return None
        page = doc.load_page(0)
        actual_rect = page.rect
        ref_a4_width, ref_a4_height = 595, 842
        scale_x = actual_rect.width / ref_a4_width
        scale_y = actual_rect.height / ref_a4_height
        ref_x0, ref_y0, ref_x1, ref_y1 = ref_rect
        target_rect = fitz.Rect(ref_x0 * scale_x, ref_y0 * scale_y, ref_x1 * scale_x, ref_y1 * scale_y)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        try:
            pix = page.get_pixmap(matrix=mat, clip=target_rect)
            pix.save("signature1.png")
            png_bytes = pix.tobytes("png")
            return base64.b64encode(png_bytes).decode("utf-8")

        except Exception as e:
            print(f"Error extracting signature region: {e}")
            return None

    def _extract_form_data_and_signature(self, pdf_bytes):
        form_data = self._extract_form_fields(pdf_bytes)
        signature_b64 = self._extract_signature_by_coordinates(pdf_bytes)
        form_data["signature"] = signature_b64
        return form_data
