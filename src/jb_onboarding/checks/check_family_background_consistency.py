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
    background = re.sub(r"\s{2,}", " ", background)
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
    match = re.search(r"\d+", children_number_sentence)
    if match:
        number = int(match.group())
    else:
        # Otherwise, try to detect if children are mentioned by name.
        if "is named" in children_number_sentence:
            number = 1
        elif "are named" in children_number_sentence:
            one_more_split = children_number_sentence.split("are named")[-1]
            number = one_more_split.count(",") + 2
        else:
            number = 0

    return {"marital_status": profile_marital_status, "number_of_children": number}


def family_background_is_consistent(data: dict):
    """
    Checks consistency between parsed family background and the client profile's marital status.
    """
    # Ensure cache directory exists
    profile = data.get("profile", {})
    description = data.get("description", {})

    if "Family Background" not in description:
        return False
    family_background = extract_family_background(description["Family Background"])

    # Get marital status from new profile structure
    profile_marital_status = profile.get("marital_status", "").strip().lower()
    if not profile_marital_status:
        return False  # Marital status missing in profile

    # Compare with extracted family background
    return profile_marital_status == family_background["marital_status"]
