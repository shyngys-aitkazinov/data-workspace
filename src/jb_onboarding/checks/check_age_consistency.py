import json
import re
from datetime import datetime
from pathlib import Path


def age_is_consistent(data: dict):
    """
    Checks if the declared age (extracted from description fields "Summary Note" or "Occupation History")
    matches the age calculated from the birth date in the profile. The profile is expected to have a top-level
    "birth_date" key in ISO format "YYYY-MM-DD".

    The declared age is looked for in the description fields by matching a pattern like "(\d+)\s+year old".
    If no declared age is found, the function returns True.

    The age is calculated using a fixed current date of 2025-04-12.

    Returns:
        True if the declared age matches the calculated age or if no declared age is provided;
        False otherwise.
    """
    description = data.get("description", {})
    profile = data.get("profile", {})

    # --- Extract declared age from description ---
    declared_age = None
    for field in ["Summary Note", "Occupation History"]:
        text = description.get(field, "")
        match = re.search(r"(\d+)\s+year old", text)
        if match:
            declared_age = int(match.group(1))
            break

    print(f"Declared age: {declared_age}")

    # If no declared age is found, nothing to check; assume it's consistent.
    if declared_age is None:
        return True

    # --- Extract birth date from profile ---
    if "birth_date" in profile:
        birth_date_str = profile["birth_date"]
    else:
        return False

    try:
        birth_date = datetime.strptime(birth_date_str, "%Y-%m-%d").date()
    except Exception as e:
        print(f"Error parsing birth_date: {e}")
        return False

    # --- Calculate age using the fixed current date "2025-04-12" ---
    current_date = datetime.strptime("2025-04-12", "%Y-%m-%d").date()
    age = current_date.year - birth_date.year
    if (current_date.month, current_date.day) < (birth_date.month, birth_date.day):
        age -= 1

    print(f"Calculated age: {age}")

    return declared_age == age
