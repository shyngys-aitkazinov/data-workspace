import os
import pathlib
import re
from os import PathLike
from pathlib import Path

from jb_onboarding.constants import DOCS
from jb_onboarding.preprocessing import Preprocessor

DATA_PATH = Path(os.path.join(os.path.dirname(__file__), "data"))


def get_dataset(data_path: PathLike) -> list[tuple[int, pathlib.Path]]:
    """
    Get the dataset of client data.

    Scans `data_path` recursively for .zip files named in the format
    'client_<number>.zip'. Extracts the <number> and returns a list of
    tuples (client_number, file_path).

    Args:
        data_path (PathLike): Path to the dataset.

    Returns:
        List[Tuple[int, pathlib.Path]]: List of tuples where the first
        element is the client number (int) and the second is the
        zip file path (pathlib.Path).
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset path {data_path} does not exist.")

    paths = []
    # Regex to match files named like "client_1234.zip" and capture the integer part
    pattern = re.compile(r"^client_(\d+)\.zip$")

    for root_dir, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(".zip"):
                # Attempt to extract the client number from the filename
                match = pattern.match(file)
                if match:
                    client_number = int(match.group(1))
                    file_path = pathlib.Path(root_dir) / file
                    paths.append((client_number, file_path))
                # If filenames might vary, you could handle that here,
                # e.g. continue, log a warning, etc.

    return sorted(paths)


def open_n_parse(file_path: PathLike) -> dict:
    """
    Open and parse a file.

    This function is a placeholder for the actual implementation that
    would open and parse the file at `file_path`. The implementation
    will depend on the file format (e.g., PDF, DOCX, TXT).

    Args:
        file_path (PathLike): Path to the file to be opened and parsed.

    Returns:
        dict: Parsed content of the file.
    """
    prep = Preprocessor()
    # Assuming the Preprocessor class has a method to handle file parsing
    result = prep(
        file_path,
        DOCS,
    )
    # Placeholder for actual parsing logic
    return result


def eval():
    print("Collecting dataset...")
    dataset = get_dataset(DATA_PATH)
    print(f"Dataset collected: {len(dataset)} files.")

    print(dataset[0])


if __name__ == "__main__":
    eval()
