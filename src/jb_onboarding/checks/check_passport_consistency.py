import json
from datetime import datetime
from pathlib import Path


def passport_is_consistent(passport):
    """
    Checks consistency on the passport file using the new input format.

    Expected keys in the passport dictionary:
        "Surname": e.g. "KOZOWSKI"
        "Code": e.g. "POLSKA"
        "Passport Number": e.g. "HE9790378"
        "Given Name": e.g. "MALWINA TERESA"
        "Birth Date": e.g. "05-Jun-1959" (or "YYYY-MM-DD")
        "Citizenship": e.g. "Polish/Polskie"
        "Sex": e.g. "F"
        "Issue Date": e.g. "17-Nov-2017"
        "Expiry Date": e.g. "16-Nov-2027"
        "MRZ Line 1": e.g. "P<OLKOZOWSKI<<MALWINA TERESA"
        "MRZ Line 2": e.g. "HE9790378POLSKA590605"
    """
    # --- Convert name fields ---
    given_name = passport.get("Given Name", "").strip()
    name_parts = given_name.split()
    first_name = name_parts[0] if name_parts else ""
    middle_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""
    last_name = passport.get("Surname", "").strip()

    # --- Process country/nationality ---
    country_code = passport.get("Code", "").strip()  # We'll use this both for country and its code.
    country = country_code
    nationality = passport.get("Citizenship", "").strip()

    # --- Process passport number and gender ---
    number = passport.get("Passport Number", "").strip()
    gender = passport.get("Sex", "").strip()
    if gender == "":
        return False

    # --- Process dates ---
    birth_date_str = passport.get("Birth Date", "").strip()
    issue_date_str = passport.get("Issue Date", "").strip()
    expiry_date_str = passport.get("Expiry Date", "").strip()

    # Try ISO format first, then fallback to "%d-%b-%Y"
    try:
        birth_date_obj = datetime.strptime(birth_date_str, "%Y-%m-%d").date()
    except ValueError:
        try:
            birth_date_obj = datetime.strptime(birth_date_str, "%d-%b-%Y").date()
        except ValueError:
            return False

    try:
        issue_date_obj = datetime.strptime(issue_date_str, "%Y-%m-%d").date()
    except ValueError:
        try:
            issue_date_obj = datetime.strptime(issue_date_str, "%d-%b-%Y").date()
        except ValueError:
            return False

    try:
        expiry_date_obj = datetime.strptime(expiry_date_str, "%Y-%m-%d").date()
    except ValueError:
        try:
            expiry_date_obj = datetime.strptime(expiry_date_str, "%d-%b-%Y").date()
        except ValueError:
            return False

    # Create ISO formatted string for birth_date to extract YYMMDD.
    birth_date_iso = birth_date_obj.strftime("%Y-%m-%d")

    # --- Process MRZ lines ---
    mrz_line_0 = passport.get("MRZ Line 1", "").strip()
    mrz_line_1 = passport.get("MRZ Line 2", "").strip()

    # Compute expected MRZ Line 0.
    # Expected format: "P<" + country_code + last_name (uppercase) + "<<" + first_name (uppercase)
    # If middle name exists, add "<" + middle_name (uppercase)
    expected_mrz_line0 = "P<" + country_code + last_name.upper() + "<<" + first_name.upper()
    if middle_name != "":
        expected_mrz_line0 += "<" + middle_name.upper()
    # Pad with '<' until the length matches that of the actual MRZ line.
    while len(expected_mrz_line0) < len(mrz_line_0):
        expected_mrz_line0 += "<"
        
    # Compute expected MRZ Line 1.
    # Expected format: passport number + country_code + YYMMDD from birth date.
    expected_mrz_line1 = number + country_code + birth_date_iso[2:4] + birth_date_iso[5:7] + birth_date_iso[8:10]
    while len(expected_mrz_line1) < len(mrz_line_1):
        expected_mrz_line1 += "<"

    # Check if the computed MRZ lines match the provided ones.
    if mrz_line_0 != expected_mrz_line0 or mrz_line_1 != expected_mrz_line1:
        return False

    # --- Date ordering checks ---
    current_date_obj = datetime.strptime("2025-04-01", "%Y-%m-%d").date()
    if (birth_date_obj > issue_date_obj or issue_date_obj > expiry_date_obj or
        birth_date_obj > current_date_obj or issue_date_obj > current_date_obj):
        return False

    # --- Nationality checks against local mapping ---
    try:
        with open('country_mappings.json', 'r') as f:
            country_data = json.load(f)
    except Exception:
        return False

    if country not in country_data:
        return False

    if country_data[country][0] != country_code or country_data[country][1] != nationality:
        return False

    return True
