import json
import os
import re
from pathlib import Path


def extract_family_background(background: str):
    """
    Processes the free-text family background to extract the marital status
    and number of children.
    """
    # Normalize whitespace and split into sentences.
    background = re.sub(r'\s{2,}', ' ', background)
    background = background.split(". ")

    marital_status_sentence = background[0].strip().lower()
    if len(background) == 1:
        children_number_sentence = ""
    else:
        children_number_sentence = background[1].strip()

    if "married" in marital_status_sentence or "tied the knot" in marital_status_sentence:
        profile_marital_status = "married"
    elif "widowed" in marital_status_sentence:
        profile_marital_status = "widowed"
    elif "single" in marital_status_sentence:
        profile_marital_status = "single"
    elif "divorced" in marital_status_sentence:
        profile_marital_status = "divorced"
    else:
        profile_marital_status = "unknown"
        children_number_sentence = marital_status_sentence

    # If there is a number, that indicates how many children there are.
    match = re.search(r'\d+', children_number_sentence)
    if match:
        number = int(match.group())
    else:
        # Otherwise, try to detect if children are mentioned by name.
        if "is named" in children_number_sentence: 
            number = 1
        elif "are named" in children_number_sentence:
            one_more_split = children_number_sentence.split("are named")[-1]
            number = one_more_split.count(',') + 2
        else:
            number = 0

    return {"marital_status": profile_marital_status, "number_of_children": number}


def family_background_is_consistent(description, profile, cache_dir_path, client_id):
    """
    Parses the family background description (from description["Family Background"]),
    storing the marital status and number of children on file, then checks if the
    client's marital status in their profile is consistent.

    New assumptions:
      - The description dictionary contains a "Family Background" key.
      - The profile dictionary now includes marital status under:
            profile["Account Holder – Personal Info"]["Marital Status"]
        which is typically structured as a list whose first element is a dictionary
        containing an entry "selected" (e.g., ["Divorced"]).
    """
    # Ensure a cache directory exists for this client.
    cur_dir_path = cache_dir_path / client_id
    os.makedirs(cur_dir_path, exist_ok=True)

    family_bckg_file_path = cur_dir_path / "family_background.json"
    if os.path.exists(family_bckg_file_path):
        family_background = json.load(family_bckg_file_path.open("r", encoding="utf-8"))
    else:
        # Expect the description to have a "Family Background" field.
        if "Family Background" not in description:
            return False
        family_background = extract_family_background(description["Family Background"])
        with open(family_bckg_file_path, "w", encoding="utf-8") as f:
            json.dump(family_background, f, indent=4)

    # --- Extract marital status from the profile ---
    # In the new input, marital status is stored under "Account Holder – Personal Info" as "Marital Status".
    marital_info = profile.get("Account Holder – Personal Info", {}).get("Marital Status", None)
    if not marital_info:
        return False

    profile_marital_status = None
    if isinstance(marital_info, list) and len(marital_info) > 0 and isinstance(marital_info[0], dict):
        selected = marital_info[0].get("selected", [])
        if selected:
            profile_marital_status = selected[0].strip().lower()
    if profile_marital_status is None:
        # Fallback: if marital_info is not in the expected structure, convert to string.
        profile_marital_status = str(marital_info).strip().lower()

    # Compare the profile marital status with that extracted from the description.
    if profile_marital_status != family_background["marital_status"]:
        return False
    return True
