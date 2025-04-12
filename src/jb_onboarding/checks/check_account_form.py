import json
from pathlib import Path


def check_name_consistency_account(data):
    """
    Checks that the full name in the account dictionary is consistent.

    Expected keys:
      - "account_name": the full name as a single string
      - "account_holder_name": the first (and possibly middle) names
      - "account_holder_surname": the last name
    For example, if:
        account_holder_name = "Joona Onni"
        account_holder_surname = "Niskanen"
    then the expected full name is: "Joona Onni Niskanen".
    """
    full_name = " ".join(data.get("account_name", "").split()).strip()
    first_middle = data.get("account_holder_name", "").strip()
    last = data.get("account_holder_surname", "").strip()

    # Build the expected full name. (We assume account_holder_name may already contain first and middle.)
    expected_full_name = f"{first_middle} {last}".strip()
    return full_name == expected_full_name


def check_email_account(data):
    """
    Checks that the email field contains an '@' symbol.
    Expected key: "email"
    """
    return "@" in data.get("email", "")


def is_single_domicile_account(data):
    """
    Checks that the domicile information is a single entry.

    In our account dictionary, we use the "country" field.
    If multiple countries were provided (separated by commas),
    this function returns False.
    """
    domicile = data.get("country", "").strip()
    return len(domicile.split(",")) == 1


def check_address_zip_account(data):
    """
    Checks that the postal code is non-empty.

    In our account dictionary the postal code is in "postal_code".
    """
    return data.get("postal_code", "").strip() != ""


def account_form_is_consistent(account_form):
    """
    Aggregates several checks for the account form using the adapted functions:
      - Name consistency (using account_name, account_holder_name, and account_holder_surname)
      - Email validity (using email)
      - Single domicile (based on the country field)
      - Postal code is provided
    """
    consistent_name = check_name_consistency_account(account_form)
    correct_email = check_email_account(account_form)
    single_domicile = is_single_domicile_account(account_form)
    address_zip = check_address_zip_account(account_form)

    return consistent_name and correct_email and single_domicile and address_zip
