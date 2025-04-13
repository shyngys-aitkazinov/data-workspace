import json
import re
from datetime import datetime
from pathlib import Path

from jb_onboarding.constants import COUNTRIES


def passport_is_consistent(data: dict, max_levenshtein: int = 5) -> bool:
    passport = data.get("passport", {})

    # --- Validate core fields exist ---
    required_fields = [
        "Country",
        "Surname",
        "Given Name",
        "Code",
        "Citizenship",
        "Passport Number",
        "Birth Date",
        "Issue Date",
        "Expiry Date",
        "MRZ Line 1",
        "MRZ Line 2",
    ]
    if any(field not in passport for field in required_fields):
        return False

    # --- Name processing ---
    last_name = passport["Surname"].strip().upper()
    given_name = passport["Given Name"].strip().upper()

    # --- Country validation ---
    country_code = passport["Code"].strip().upper()
    nationality = passport["Citizenship"].split("/")[0].strip().title()
    country = passport["Country"].strip()
    passport_number = passport["Passport Number"].strip().upper()
    # Validate country code and nationality match
    try:
        country_data = COUNTRIES[country]
        if country_data[0] != country_code:  # Compare alpha-3 codes
            return False
        if country_data[1] != nationality:
            return False
    except KeyError:
        return False
    # --- MRZ Line 1 processing ---

    mrz_together = passport["MRZ Line 1"].upper() + passport["MRZ Line 2"].upper()
    mrz_together = re.sub(r"[<>\s]", "", mrz_together)
    # Date parsing and formatting
    try:
        birth_date = datetime.strptime(passport["Birth Date"], "%d-%b-%Y")
    except ValueError:
        return False

    # Build expected MRZ components
    expected_start = f"{passport_number}{country_code}{birth_date.strftime('%y%m%d')}"

    # --- Date validity checks ---
    try:
        issue_date = datetime.strptime(passport["Issue Date"], "%d-%b-%Y").date()

        expiry_date = datetime.strptime(passport["Expiry Date"], "%d-%b-%Y").date()
    except ValueError:
        return False

    today = datetime.today().date()

    return all([birth_date.date() < issue_date, issue_date <= expiry_date, expiry_date > today])
