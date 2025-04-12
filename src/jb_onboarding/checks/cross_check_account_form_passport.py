import json
from pathlib import Path

from tqdm import tqdm


def account_form_and_passport_are_consistent(account_form, passport):
    # --- Extract and normalize name parts from the account form ---
    # Try using "name", falling back to "account_name"
    full_name_acc = account_form.get("name", account_form.get("account_name", "")).strip()
    name_parts_acc = full_name_acc.split()
    # Assume: first token -> first_name, last token -> last_name, remaining tokens (if any) are middle_name.
    if len(name_parts_acc) >= 2:
        acc_first = name_parts_acc[0].lower().strip()
        acc_last = name_parts_acc[-1].lower().strip()
        if len(name_parts_acc) > 2:
            acc_middle = " ".join(name_parts_acc[1:-1]).lower().strip()
        else:
            acc_middle = ""
    else:
        # Not enough parts to form a full name
        return False

    # --- Extract and normalize name parts from the passport ---
    # For the passport, the first name and possibly middle names are in "Given Name",
    # and the last name is in "Surname".
    given_name_pass = passport.get("Given Name", "").strip()
    name_parts_pass = given_name_pass.split()
    if len(name_parts_pass) >= 1:
        pass_first = name_parts_pass[0].lower().strip()
        if len(name_parts_pass) > 1:
            pass_middle = " ".join(name_parts_pass[1:]).lower().strip()
        else:
            pass_middle = ""
    else:
        return False

    pass_last = passport.get("Surname", "").lower().strip()

    # --- Compare the corresponding name parts ---
    if acc_first != pass_first:
        return False
    if acc_middle != pass_middle:
        return False
    if acc_last != pass_last:
        return False

    # --- Normalize passport numbers ---
    acc_passport = account_form.get("passport_number", "")
    # If the account form passport number is given as a list, take its first element.
    if isinstance(acc_passport, list):
        acc_passport = acc_passport[0]
    acc_passport = acc_passport.strip()

    pass_passport = passport.get("Passport Number", "").strip()

    if acc_passport != pass_passport:
        return False

    return True