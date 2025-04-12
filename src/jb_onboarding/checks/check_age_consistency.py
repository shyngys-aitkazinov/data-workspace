import json
import re
from datetime import datetime
from pathlib import Path


def age_is_consistent(description, profile):
    """
    Checks if the declared age (found in description fields "Summary Note" or "Occupation History")
    matches the age calculated from the birth date in profile using the fixed current date 2025-04-01.

    For profile:
      - The birth date is expected to be found at profile["Client Information"]["Date of birth"].
      - (If not found there, an alternate key "birth_date" is also checked.)
      
    For description:
      - The function looks for a pattern like "(\d+)\s+year old" in the text
        of either the "Summary Note" or "Occupation History" fields.
      - If no declared age is found, the check returns True.
    """
    # --- Extract declared age from description ---
    declared_age = None
    # Try keys in description in order.
    for field in ["Summary Note", "Occupation History"]:
        text = description.get(field, "")
        match = re.search(r"(\d+)\s+year old", text)
        if match:
            declared_age = int(match.group(1))
            break

    # If no declared age is found, nothing to check:
    if declared_age is None:
        return True

    # --- Extract actual birth date from profile ---
    try:
        # First, try to extract from the nested Client Information section.
        if "Client Information" in profile and "Date of birth" in profile["Client Information"]:
            birth_date_str = profile["Client Information"]["Date of birth"]
        elif "birth_date" in profile:
            birth_date_str = profile["birth_date"]
        else:
            return False

        # Expecting ISO format YYYY-MM-DD.
        birth_date = datetime.strptime(birth_date_str, "%Y-%m-%d").date()
    except Exception:
        return False

    # --- Calculate age based on fixed current date ---
    current_date = datetime.strptime("2025-04-01", "%Y-%m-%d").date()
    age = current_date.year - birth_date.year
    if (current_date.month, current_date.day) < (birth_date.month, birth_date.day):
        age -= 1

    return declared_age == age
