import json
import re
from datetime import datetime
from pathlib import Path

from jb_onboarding.constants import COUNTRIES


def passport_is_consistent(data: dict) -> bool:
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
    mrz_line1 = passport["MRZ Line 1"].upper()

    # Validate MRZ structure
    if not mrz_line1.startswith(f"P<{country_code}"):
        return False

    # Extract name components using any combination of < or >
    mrz_parts = re.split(r"[<>]+", mrz_line1[5:])  # Skip P< + country_code
    mrz_parts = [p.strip() for p in mrz_parts if p.strip()]
    if not mrz_parts:
        return False

    # Compare names with passport data
    mrz_surname = mrz_parts[0]
    mrz_given_names = " ".join(mrz_parts[1:]) if len(mrz_parts) > 1 else ""
    if last_name != mrz_surname or given_name != mrz_given_names:
        return False

    # --- MRZ Line 2 validation ---
    mrz_line2 = passport["MRZ Line 2"].upper()
    passport_number = passport["Passport Number"].strip().upper()

    # Date parsing and formatting
    try:
        birth_date = datetime.strptime(passport["Birth Date"], "%d-%b-%Y")
    except ValueError:
        return False

    # Build expected MRZ components
    expected_start = f"{passport_number}{country_code}{birth_date.strftime('%y%m%d')}"
    if not mrz_line2.startswith(expected_start):
        return False

    # --- Date validity checks ---
    try:
        issue_date = datetime.strptime(passport["Issue Date"], "%d-%b-%Y").date()

        expiry_date = datetime.strptime(passport["Expiry Date"], "%d-%b-%Y").date()
    except ValueError:
        return False

    today = datetime.today().date()

    return all([birth_date.date() < issue_date, issue_date <= expiry_date, expiry_date > today])
