import re


def extract_account_currency(account_form):
    """
    Determine the currency from the account form.
    If a key "currency" exists and is nonempty, use that. Otherwise, check flags.
    Assumptions:
      - If account_form["eur"] equals "/Yes", return "EUR";
      - if account_form["usd"] equals "/Yes", return "USD";
      - if account_form["chf"] equals "/Yes", return "CHF";
      - otherwise, return an empty string.
    """
    currency = account_form.get("currency", "").strip()
    if currency:
        return currency
    else:
        if account_form.get("eur", "").strip() == "/Yes":
            return "EUR"
        elif account_form.get("usd", "").strip() == "/Yes":
            return "USD"
        elif account_form.get("chf", "").strip() == "/Yes":
            return "CHF"
        else:
            return ""


def build_account_address(account_form):
    """
    Combine separate address fields in the account form into a single normalized string.
    Expected keys: "building_number", "street_name", "postal_code", "city"
    For example, given:
       building_number: '28'
       street_name: 'Enzersdorfer Straße'
       postal_code: '4503'
       city: 'Bruck an der Mur'
    The output is: "Enzersdorfer Straße 28, 4503 Bruck an der Mur".
    """
    building = account_form.get("building_number", "").strip()
    street = account_form.get("street_name", "").strip()
    postal = account_form.get("postal_code", "").strip()
    city = account_form.get("city", "").strip()
    parts = []
    if street:
        parts.append(f"{street} {building}" if building else street)
    if postal or city:
        parts.append(f"{postal} {city}".strip())
    return ", ".join(parts).strip()


def extract_profile_full_name(client_profile):
    """
    In the new profile structure, the full name is provided at the top level.
    """
    return client_profile.get("name", "").strip()


def extract_profile_address(client_profile):
    """
    Converts the address dictionary in the profile into a normalized string.
    Expected keys in the address dict: "street name", "street number", "postal code", "city".
    For example, it returns "Enzersdorfer Straße 28, 4503 Bruck an der Mur".
    """
    addr = client_profile.get("address", {})
    street = addr.get("street name", "").strip()
    number = addr.get("street number", "").strip()
    postal = addr.get("postal code", "").strip()
    city = addr.get("city", "").strip()
    parts = []
    if street:
        parts.append(f"{street} {number}" if number else street)
    if postal or city:
        parts.append(f"{postal} {city}".strip())
    return ", ".join(parts).strip()


def extract_profile_country_of_domicile(client_profile):
    """
    In the new profile structure, country of domicile is stored in the top-level key "country_of_domicile".
    """
    return client_profile.get("country_of_domicile", "").strip()


def extract_profile_passport_number(client_profile):
    """
    In the new profile structure, the passport number is stored as a top-level key "passport_number".
    """
    return client_profile.get("passport_number", "").strip()


def extract_profile_contact_info(client_profile):
    """
    In the new profile structure, the phone number and email address are stored as top-level keys.
    Returns a tuple: (phone_number, email_address).
    """
    phone = client_profile.get("phone_number", "").strip()
    email = client_profile.get("email_address", "").strip()
    return phone, email


def account_form_and_client_profile_are_consistent(data: dict) -> bool:
    """
    Checks consistency between the account form and the client profile.

    From the account form (new format), expected keys include:
       - "name" or "account_name": the full name.
       - "passport_number"
       - Currency: either a direct "currency" key (if nonempty) or derived from flags.
       - Address: built from "building_number", "street_name", "postal_code", "city".
       - "country": country of domicile.
       - "phone_number"
       - "email"

    From the client profile, expected keys include:
       - "name": the full name.
       - "address": a dictionary with keys "street name", "street number", "postal code", "city".
       - "country_of_domicile"
       - "passport_number"
       - "phone_number" and "email_address"
       - "currency": if not provided, assume it should match the account form’s derived currency.

    Additionally, the phone number check tolerates formatting differences (i.e. one number must be a substring of the other).
    """
    profile = data.get("profile", {})
    account = data.get("account", {})
    # --- Extract fields from account form ---
    acc_name = account.get("name", account.get("account_name", "")).strip()
    acc_passport = account.get("passport_number", "").strip()
    acc_currency = account.get("currency", "").strip() or extract_account_currency(account)
    acc_address = build_account_address(account)
    acc_country = account.get("country", "").strip()  # 'country' key in account form holds the domicile.
    acc_phone = account.get("phone_number", "").strip()
    acc_email = account.get("email", "").strip()

    # --- Extract fields from client profile ---
    profile_name = extract_profile_full_name(profile)
    profile_address = extract_profile_address(profile)
    profile_country = extract_profile_country_of_domicile(profile)
    profile_passport = extract_profile_passport_number(profile)
    profile_phone, profile_email = extract_profile_contact_info(profile)
    profile_currency = profile.get("currency", "").strip() or acc_currency

    # --- Compare the fields ---
    if acc_name != profile_name or acc_passport != profile_passport:
        return False

    if acc_country != profile_country or acc_currency != profile_currency:
        return False

    if acc_email != profile_email or acc_address != profile_address:
        return False

    # For phone numbers, allow for formatting differences.
    if profile_phone not in acc_phone and acc_phone not in profile_phone:
        return False

    return True
