import json
from pathlib import Path


def client_profile_and_passport_are_consistent(client_profile, passport):
    # --- Extract passport data ---
    # Passport keys in the new format: "Given Name" and "Surname".
    given_name = passport.get("Given Name", "").strip()
    name_parts = given_name.split()
    passport_first_name = name_parts[0] if name_parts else ""
    passport_middle_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""
    passport_last_name = passport.get("Surname", "").strip()

    # Construct full name variations:
    if passport_middle_name:
        passport_full_name = f"{passport_first_name} {passport_middle_name} {passport_last_name}".strip()
    else:
        passport_full_name = f"{passport_first_name} {passport_last_name}".strip()
    # Here, passport_full_name_no_space is simply the same (we already trimmed spaces)
    passport_full_name_no_space = passport_full_name

    # Other passport fields:
    passport_birth_date_str = passport.get("Birth Date", "").strip()
    passport_gender = passport.get("Sex", "").strip()
    passport_nationality = passport.get("Citizenship", "").strip()
    passport_number = passport.get("Passport Number", "").strip()
    passport_issue_date_str = passport.get("Issue Date", "").strip()
    passport_expiry_date_str = passport.get("Expiry Date", "").strip()

    # --- Extract client profile data from "Client Information" ---
    client_info = client_profile.get("Client Information", {})
    profile_first_middle = client_info.get("First/ Middle Name (s)", "").strip()
    profile_last = client_info.get("Last Name", "").strip()
    profile_full_name = f"{profile_first_middle} {profile_last}".strip()

    profile_birth_date_str = client_info.get("Date of birth", "").strip()
    profile_nationality = client_info.get("Nationality", "").strip()
    # Extract profile gender: expect it in a dictionary under key "Gender"
    gender_info = client_info.get("Gender", {})
    if isinstance(gender_info, dict):
        selected = gender_info.get("selected", [])
        profile_gender = selected[0].strip() if selected else ""
    else:
        profile_gender = str(gender_info).strip()

    profile_passport_number = client_info.get("Passport No/ Unique ID", "").strip()
    profile_issue_date_str = client_info.get("ID Issue Date", "").strip()
    profile_expiry_date_str = client_info.get("ID Expiry Date", "").strip()

    # --- Check for consistency ---
    # Check full name: we compare both the version with spaces and a re-trimmed version.
    if passport_full_name != profile_full_name and passport_full_name_no_space != profile_full_name:
        return False

    if passport_birth_date_str != profile_birth_date_str:
        return False
    if passport_gender != profile_gender:
        return False
    if passport_nationality != profile_nationality:
        return False
    if passport_number != profile_passport_number:
        return False
    if passport_issue_date_str != profile_issue_date_str or passport_expiry_date_str != profile_expiry_date_str:
        return False

    return True
