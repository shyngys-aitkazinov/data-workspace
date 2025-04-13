def client_profile_and_passport_are_consistent(data: dict) -> bool:
    passport = data.get("passport", {})
    profile = data.get("profile", {})
    # --- Extract passport data ---
    # Passport keys in the new format: "Given Name" and "Surname".
    given_name = passport.get("Given Name", "").strip()
    name_parts = given_name.split()
    passport_first_name = name_parts[0] if name_parts else ""
    passport_middle_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""
    passport_last_name = passport.get("Surname", "").strip()

    # Construct full name variations.
    if passport_middle_name:
        passport_full_name = f"{passport_first_name} {passport_middle_name} {passport_last_name}".strip()
    else:
        passport_full_name = f"{passport_first_name} {passport_last_name}".strip()
    # Here, passport_full_name_no_space is the same (we already trimmed extra spaces)
    passport_full_name_no_space = passport_full_name

    # Other passport fields.
    passport_birth_date_str = passport.get("Birth Date", "").strip()
    passport_gender = passport.get("Sex", "").strip()
    passport_nationality = passport.get("Citizenship", "").strip()
    passport_number = passport.get("Passport Number", "").strip()
    passport_issue_date_str = passport.get("Issue Date", "").strip()
    passport_expiry_date_str = passport.get("Expiry Date", "").strip()

    # --- Extract client profile data from the new flat structure ---
    profile_full_name = profile.get("name", "").strip()
    profile_birth_date_str = profile.get("birth_date", "").strip()
    profile_nationality = profile.get("nationality", "").strip()
    profile_gender = profile.get("gender", "").strip()
    profile_passport_number = profile.get("passport_number", "").strip()
    profile_issue_date_str = profile.get("passport_issue_date", "").strip()
    profile_expiry_date_str = profile.get("passport_expiry_date", "").strip()

    # --- Check for consistency ---
    # Compare full names.
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
