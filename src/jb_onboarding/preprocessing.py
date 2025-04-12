import os
import zipfile
from os import PathLike

from docx import Document

from jb_onboarding.constants import DOCS


class Preprocessor:
    def __init__(self):
        """
        This is the constructor method for the Preprocessor class.
        It initializes the class with the provided arguments.
        """
        pass

    def __call__(self, path_to_zip: PathLike):
        """
        This method is called when the Preprocessor class is instantiated.
        It initializes the class with the provided arguments.
        """
        with zipfile.ZipFile(path_to_zip, "r") as zip_ref:
            # extract the zip file to a temporary directory
            temp_dir = str(path_to_zip).replace(".zip", "")
            zip_ref.extractall(temp_dir)
            # check if the profile.docx file exists in the zip archive
            if "profile.docx" not in zip_ref.namelist():
                raise FileNotFoundError("profile.docx not found in the zip archive.")

            with zip_ref.open("profile.docx") as docx_file:
                profile = Document(docx_file)

            if "passport.png" not in zip_ref.namelist():
                raise FileNotFoundError("passport.png not found in the zip archive.")
            passport_path = os.path.join(temp_dir, "passport.png")
