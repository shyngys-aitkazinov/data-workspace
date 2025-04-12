import json
import os
import re
from pathlib import Path


def profile_is_consistent(profile):
    """
    Checks if graduation years (based on higher education) are logically consistent with
    the client’s birth date using the new input format.
    
    Adapted assumptions:
      - Birth date is taken from profile["Client Information"]["Date of birth"]
        (expected format: "YYYY-MM-DD").
      - Higher education graduation year is extracted from 
        profile["Account Holder – Personal Info"]["Education History"], which is a string
        that should contain a 4-digit year.
      - We check that the higher education graduation year is at least 18 years after the 
        birth year (typical minimum age for completing higher education) and at most 40 years 
        after the birth year.
        
    Note: Checks for secondary school graduation, employment history, real estate property values,
          and investment fields are omitted because the new input format does not provide these in a structured way.
    """
    try:
        # Extract birth year from "Client Information"
        birth_date_str = profile["Client Information"]["Date of birth"]
        birth_year = int(birth_date_str.split("-")[0])
    except Exception as e:
        # Required birth date field missing or not in expected format.
        return False

    # Extract higher education graduation info from "Education History" in "Account Holder – Personal Info"
    education_history = profile.get("Account Holder – Personal Info", {}).get("Education History", "")
    if not education_history:
        return False

    # Use regex to find a 4-digit year (e.g., "1989") in the education history.
    match = re.search(r"(19|20)\d{2}", education_history)
    if not match:
        return False
    higher_ed_year = int(match.group(0))

    # Check that the graduation year is at least 18 years after the birth year.
    if higher_ed_year - birth_year < 18:
        return False

    # Also check that the difference is not unreasonably large (e.g., more than 40 years).
    if higher_ed_year - birth_year > 40:
        return False

    return True