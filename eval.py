import json
import os
import pathlib
import re
from os import PathLike
from pathlib import Path
from random import seed, shuffle

from sklearn.metrics import classification_report

from jb_onboarding.constants import DOCS, default_rules
from jb_onboarding.preprocessing import Preprocessor
from jb_onboarding.validator import ClientValidator

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

    # randomly sample 50  from first 500 and 50 from 500-1000
    paths.sort(key=lambda x: x[0])
    paths = paths[:50] + paths[950:1000]
    seed(42)
    # shuffle(paths)
    return paths


def eval():
    print("Collecting dataset...")
    dataset = get_dataset(DATA_PATH)
    print(f"Dataset collected: {len(dataset)} files.")

    print(dataset[0])
    prep = Preprocessor()
    evaluator = ClientValidator(default_rules)

    preds = []
    labels = []

    for client_id, dataset_item in dataset:
        print(f"Processing client {client_id}...")
        # Open and parse the file
        client_data, flag = prep(dataset_item)
        # client_data = json.loads(client_data)

        with open("my_json.json", "w", encoding="utf-8") as f:
            json.dump(client_data, f, ensure_ascii=False, indent=2)

        pred = evaluator(client_data, flag=flag)
        preds.append(int(pred))
        label = 1 if client_id < 500 else 0
        labels.append(label)
        print(f"Client {client_id} prediction: {pred}, expected: {label}")

    # Calculate accuracy
    print(classification_report(labels, preds, target_names=["Accept", "Reject"]))


if __name__ == "__main__":
    eval()
