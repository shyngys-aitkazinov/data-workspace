import json
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


def eval():
    print("Collecting dataset...")
    dataset = get_dataset(DATA_PATH)
    print(f"Dataset collected: {len(dataset)} files.")

    print(dataset[0])
    prep = Preprocessor()

    for client_id, dataset_item in dataset:
        if client_id == 1:
            continue
        print(f"Processing client {client_id}...")
        # Open and parse the file
        client_data = prep(dataset_item)

        print(client_data)

        with open(f"{client_id}.json", "w") as f:
            # Assuming client_data is a dictionary or list that can be serialized to JSON
            json.dump(client_data, f, indent=4)

        break

        # Placeholder for further processing of parsed data
        print(f"Parsed data for client {client_id}: {parsed_data}")


if __name__ == "__main__":
    eval()
