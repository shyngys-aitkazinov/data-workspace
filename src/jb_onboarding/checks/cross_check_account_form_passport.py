import re


def account_form_and_passport_are_consistent(data: dict) -> bool:
    """
    Checks consistency between the account form and the passport information.

    New assumptions for the account form (new input structure):
       - The account form may include separate keys "account_holder_name" and "account_holder_surname".
         If they exist, these are used to derive the first and last names. Otherwise, we fall back
         to splitting the "name" or "account_name" field.
       - The passport number in the account form is given under "passport_number".

    Passport information is expected to have:
       - "Given Name": may include first name and middle names.
       - "Surname": the last name.
       - "Passport Number": the passport's number.

    The function returns False if:
       - The first token of the account form’s first name does not match the first token of the
         passport’s "Given Name".
       - The account form's last name does not match the passport's "Surname".
       - The passport number in the account form does not equal the passport's "Passport Number".
    Otherwise, it returns True.
    """
    account = data.get("account", {})
    passport = data.get("passport", {})
    # --- Extract name parts from the account form ---
    # Prefer to use the keys "account_holder_name" and "account_holder_surname" if available.
    account_holder_name = account.get("account_holder_name", "").strip()
    account_holder_surname = account.get("account_holder_surname", "").strip()

    if account_holder_name and account_holder_surname:
        # Use the first token of the account_holder_name as the first name.
        acc_first = account_holder_name.split()[0].lower()
        acc_last = account_holder_surname.lower()
    else:
        # Fall back to splitting "name" (or "account_name")
        full_name = account.get("name", account.get("account_name", "")).strip()
        name_parts = full_name.split()
        if len(name_parts) < 2:
            return False  # Not enough parts to form a valid first and last name.
        acc_first = name_parts[0].lower()
        acc_last = name_parts[-1].lower()

    # --- Extract name parts from the passport ---
    # Passport: first name(s) come from "Given Name" and last name from "Surname"
    given_name_pass = passport.get("Given Name", "").strip().lower()
    if not given_name_pass:
        return False
    pass_first = given_name_pass.split()[0]  # Compare only the first token.
    pass_last = passport.get("Surname", "").strip().lower()

    # --- Compare names ---
    if acc_first != pass_first:
        return False
    if acc_last != pass_last:
        return False

    # --- Compare passport numbers ---
    # If the account passport number is a list, take the first element.
    acc_passport = account.get("passport_number", "")
    if isinstance(acc_passport, list):
        acc_passport = acc_passport[0]
    acc_passport = acc_passport.strip()

    pass_passport = passport.get("Passport Number", "").strip()

    if acc_passport != pass_passport:
        return False

    return True
