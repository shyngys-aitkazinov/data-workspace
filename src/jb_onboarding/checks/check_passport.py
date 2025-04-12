import json
from datetime import datetime


def passport_is_consistent(passport):
    """
    Checks consistency of the passport file using the raw parsed data.
    The input passport dictionary is assumed to have these keys:
        "Surname", "Code", "Passport Number", "Given Name", "Birth Date",
        "Citizenship", "Sex", "Issue Date", "Expiry Date", "MRZ Line 1", "MRZ Line 2"

    The function uses a relaxed approach by:
      - Splitting MRZ Line 1 on the delimiter '<<' and comparing the surname and given names.
      - Removing filler characters from MRZ Line 2 and checking for the passport number,
        issuing country (as provided in "Code"), and the birth date (formatted as YYMMDD).
      - Performing date ordering checks and verifying the country details via a local JSON mapping.
    """

    # --- Extract values from the passport dictionary ---
    last_name = passport.get("Surname", "").strip()
    code = passport.get("Code", "").strip()  # Issuing country (e.g., "POLSKA")
    passport_number = passport.get("Passport Number", "").strip()
    given_name = passport.get("Given Name", "").strip()  # May include first and middle names
    birth_date_str = passport.get("Birth Date", "").strip()
    citizenship = passport.get("Citizenship", "").strip()  # Expected nationality mapping
    gender = passport.get("Sex", "").strip()
    issue_date_str = passport.get("Issue Date", "").strip()
    expiry_date_str = passport.get("Expiry Date", "").strip()
    mrz_line1 = passport.get("MRZ Line 1", "").strip()
    mrz_line2 = passport.get("MRZ Line 2", "").strip()

    # --- Basic presence checks ---
    if not (gender and mrz_line1 and mrz_line2):
        return False

    # --- Process MRZ Line 1 (names and issuing info) ---
    mrz_line1_upper = mrz_line1.upper()
    # The MRZ line should start with "P<"
    if not mrz_line1_upper.startswith("P<"):
        return False
    # Split the line using the standard name separator "<<"
    parts = mrz_line1_upper.split("<<")
    if len(parts) < 2:
        return False

    # The first part: remove the initial "P<" then remove any remaining filler
    surname_mrz = parts[0][2:].replace("<", "")
    # Compare the provided surname against the MRZ surname
    if last_name.upper() not in surname_mrz:
        return False

    # The second part should contain the given names.
    # Replace filler characters with spaces and collapse multiple spaces.
    given_names_mrz = " ".join(parts[1].replace("<", " ").split())
    if given_name.upper() not in given_names_mrz:
        return False

    # --- Process MRZ Line 2 (passport number, country code, birth date) ---
    mrz_line2_clean = mrz_line2.upper().replace("<", "")
    # Check that the passport number appears in MRZ Line 2
    if passport_number.upper() not in mrz_line2_clean:
        return False
    # Check that the issuing code (e.g., "POLSKA") appears in MRZ Line 2
    if code.upper() not in mrz_line2_clean:
        return False

    # Parse the birth date to create a YYMMDD string.
    try:
        # Try ISO format first ("YYYY-MM-DD")
        birth_date_obj = datetime.strptime(birth_date_str, "%Y-%m-%d").date()
    except ValueError:
        try:
            # Try abbreviated format (e.g., "05-Jun-1959")
            birth_date_obj = datetime.strptime(birth_date_str, "%d-%b-%Y").date()
        except ValueError:
            return False
    birth_date_yymmdd = birth_date_obj.strftime("%y%m%d")
    if birth_date_yymmdd not in mrz_line2_clean:
        return False

    # --- Check date ordering ---
    try:
        try:
            issue_date = datetime.strptime(issue_date_str, "%Y-%m-%d").date()
        except ValueError:
            issue_date = datetime.strptime(issue_date_str, "%d-%b-%Y").date()

        try:
            expiry_date = datetime.strptime(expiry_date_str, "%Y-%m-%d").date()
        except ValueError:
            expiry_date = datetime.strptime(expiry_date_str, "%d-%b-%Y").date()
    except Exception:
        return False

    if birth_date_obj > issue_date or issue_date > expiry_date:
        return False

    # Use a fixed current date for consistency (you may use datetime.today() if desired)
    current_date = datetime.strptime("2025-04-01", "%Y-%m-%d").date()
    if issue_date > current_date:
        return False

    # --- Verify country and nationality using a local JSON mapping ---
    try:
        with open('country_mappings.json', 'r') as f:
            country_data = json.load(f)
    except Exception:
        return False

    # Here we assume the mapping uses the "Citizenship" field as key and provides:
    # country_data[citizenship] = [expected_code, expected_nationality]
    if citizenship not in country_data:
        return False
    if country_data[citizenship][0].upper() != code.upper():
        return False
    if country_data[citizenship][1].upper() != citizenship.upper():
        return False

    return True
