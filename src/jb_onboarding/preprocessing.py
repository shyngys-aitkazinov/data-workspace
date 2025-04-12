import base64
import json
import os
import re
import zipfile
from io import BytesIO

import fitz
import matplotlib.pyplot as plt
from docx import Document
from PIL import Image
from PyPDF2 import PdfReader


class Preprocessor:
    def __init__(self):
        self.profile_json = None
        self.passport_placeholder = None
        self.description_json = None
        self.account_pdf_bytes = None
        self.account_parsed = None

    def __call__(self, path_to_zip):
        temp_dir = str(path_to_zip).replace(".zip", "")
        with zipfile.ZipFile(path_to_zip, "r") as zip_ref:
            zip_ref.extractall(temp_dir)
            names = zip_ref.namelist()
            if "profile.docx" not in names:
                raise FileNotFoundError("profile.docx not found in the zip archive.")
            with zip_ref.open("profile.docx") as profile_file:
                doc = Document(profile_file)
                self.profile_json = self._parse_profile_docx_to_json(doc)
            if "passport.png" not in names:
                raise FileNotFoundError("passport.png not found in the zip archive.")
            self.passport_placeholder = "passport logic placeholder"
            if "description.txt" not in names:
                raise FileNotFoundError("description.txt not found in the zip archive.")
            with zip_ref.open("description.txt") as desc_file:
                raw_desc = desc_file.read().decode("utf-8")
                self.description_json = self._parse_description(raw_desc)
            if "account.pdf" not in names:
                raise FileNotFoundError("account.pdf not found in the zip archive.")
            with zip_ref.open("account.pdf") as pdf_file:
                self.account_pdf_bytes = pdf_file.read()
        self.account_parsed = self._extract_form_data_and_signature(self.account_pdf_bytes)
        output = {
            "profile": self.profile_json,
            "passport": self.passport_placeholder,
            "description": self.description_json,
            "account": self.account_parsed
        }
        return json.dumps(output, indent=4, ensure_ascii=False)

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
        return ' '.join(text.replace("\t", " ").split())

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
                if not key:
                    continue
                value_candidates = [cell for cell in row[1:] if cell]
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
                value = field_info.get('/V', '')
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