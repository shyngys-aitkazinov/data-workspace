from datetime import datetime


def client_profile_and_passport_are_consistent(data: dict) -> bool:
    """Check consistency between client profile and passport data with new structure"""
    profile = data.get("profile", {})
    passport = data.get("passport", {})

    # Name comparison
    profile_name = profile.get("name", "").strip().upper()
    passport_surname = passport.get("Surname", "").strip().upper()
    passport_given = passport.get("Given Name", "").strip().upper()

    # Split profile name into components
    profile_parts = profile_name.split()
    passport_parts = f"{passport_given} {passport_surname}".split()

    # Compare all name components regardless of order
    if sorted(profile_parts) != sorted(passport_parts):
        return False

    # Date comparison with format normalization
    def normalize_date(date_str):
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            try:
                return datetime.strptime(date_str, "%d-%b-%Y").date()
            except:
                return None

    # Birth date
    profile_dob = normalize_date(profile.get("birth_date", ""))
    passport_dob = normalize_date(passport.get("Birth Date", ""))
    if profile_dob != passport_dob:
        return False

    # Gender comparison
    gender_mapping = {"M": "MALE", "F": "FEMALE"}
    profile_gender = gender_mapping.get(profile.get("gender", "").upper(), "")
    passport_gender = passport.get("Sex", "").upper()
    if profile_gender and passport_gender:
        if profile_gender not in passport_gender and passport_gender not in profile_gender:
            return False

    # Nationality comparison
    profile_nationality = profile.get("nationality", "").upper()
    passport_nationality = passport.get("Citizenship", "").split("/")[0].strip().upper()
    if profile_nationality != passport_nationality:
        return False

    # Passport number comparison
    if profile.get("passport_number", "").upper() != passport.get("Passport Number", "").upper():
        return False

    # Date validity check
    today = datetime.today().date()
    expiry = normalize_date(passport.get("Expiry Date", ""))
    if not expiry or expiry <= today:
        return False

    return True
