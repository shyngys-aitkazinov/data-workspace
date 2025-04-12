import json
from pathlib import Path


def extract_account_currency(account_form):
    """
    Determine the currency from the account form.
    If a key "currency" exists, use that. Otherwise, check flags.
    Assumption: if 'eur' equals "/Yes", return "EUR"; if 'usd' equals "/Yes", return "USD";
    if 'chf' equals "/Yes", return "CHF"; otherwise, return an empty string.
    """
    if "currency" in account_form:
        return account_form["currency"].strip()
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
    """
    building = account_form.get("building_number", "").strip()
    street = account_form.get("street_name", "").strip()
    postal = account_form.get("postal_code", "").strip()
    city = account_form.get("city", "").strip()
    # Build address string. If any part is missing, they will be skipped.
    parts = []
    if street:
        if building:
            parts.append(f"{street} {building}")
        else:
            parts.append(street)
    if postal or city:
        parts.append(f"{postal} {city}".strip())
    return ", ".join(parts).strip()

def extract_profile_full_name(client_profile):
    """
    Construct the full name from the client profile's "Client Information" section.
    Expected keys: "First/ Middle Name (s)" and "Last Name".
    """
    client_info = client_profile.get("Client Information", {})
    first_middle = client_info.get("First/ Middle Name (s)", "").strip()
    last = client_info.get("Last Name", "").strip()
    full_name = f"{first_middle} {last}".strip()
    return full_name

def extract_profile_address(client_profile):
    """
    Extract the address from the client profile's "Client Information" section.
    Expected key: "Address".
    """
    client_info = client_profile.get("Client Information", {})
    return client_info.get("Address", "").strip()

def extract_profile_country_of_domicile(client_profile):
    """
    Extract country of domicile from the client profile's "Client Information" section.
    Expected key: "Country of Domicile".
    """
    client_info = client_profile.get("Client Information", {})
    return client_info.get("Country of Domicile", "").strip()

def extract_profile_passport_number(client_profile):
    """
    Extract the passport number from the client profile's "Client Information" section.
    Expected key: "Passport No/ Unique ID".
    """
    client_info = client_profile.get("Client Information", {})
    return client_info.get("Passport No/ Unique ID", "").strip()

def extract_profile_contact_info(client_profile):
    """
    Extract the phone number and email address from the profile's contact info section.
    Expected section: "Account Holder – Contact Management and Services – Contact Info".
    
    We assume the phone number is found in the value corresponding to "Communication Medium",
    which typically starts with "Telephone", and the email address in a field that contains "E-Mail".
    """
    contact = client_profile.get("Account Holder – Contact Management and Services – Contact Info", {})
    phone_raw = contact.get("Communication Medium", "").strip()
    # Remove known keywords like "Telephone" and extra whitespace.
    phone_number = phone_raw.replace("Telephone", "").strip()
    
    # Find an email address: if a key such as "null" exists containing a string starting with "E-Mail", extract it.
    email_raw = ""
    for key, value in contact.items():
        if isinstance(value, str) and "E-Mail" in value:
            email_raw = value.replace("E-Mail", "").strip()
            break
    return phone_number, email_raw

def account_form_and_client_profile_are_consistent(account_form, client_profile):
    """
    Checks consistency between the account form and the client profile.
    
    From the account form, expected keys (new format):
       - "name" (or "account_name"): full name.
       - "passport_number"
       - Currency: either from "currency" key or derived from flags ("eur", "usd", "chf").
       - Address: built from "building_number", "street_name", "postal_code", "city".
       - "country_of_domicile" is taken from "country".
       - "phone_number"
       - "email" (as email address)
    
    From the client profile, expected fields in "Client Information" and contact info:
       - Full name: from "First/ Middle Name (s)" and "Last Name"
       - Address: from "Address" (a string)
       - Country of domicile: from "Country of Domicile"
       - Passport number: from "Passport No/ Unique ID"
       - Contact info (phone number and email): from the "Account Holder – Contact Management and Services – Contact Info" section.
       - Currency: Not explicitly provided; assume it should match the account form's currency.
       
    Additionally, the phone number check is tolerant regarding formatting:
       The function returns False if neither value appears as a substring in the other.
    """
    # --- Extract fields from the account form ---
    acc_name = account_form.get("name", account_form.get("account_name", "")).strip()
    acc_passport = account_form.get("passport_number", "").strip()
    acc_currency = account_form.get("currency", "").strip() or extract_account_currency(account_form)
    acc_address = build_account_address(account_form)
    acc_country = account_form.get("country", "").strip()  # Using "country" as domicile.
    acc_phone = account_form.get("phone_number", "").strip()
    acc_email = account_form.get("email", "").strip()

    # --- Extract fields from the client profile ---
    profile_name = extract_profile_full_name(client_profile)
    profile_address = extract_profile_address(client_profile)
    profile_country = extract_profile_country_of_domicile(client_profile)
    profile_passport = extract_profile_passport_number(client_profile)
    profile_phone, profile_email = extract_profile_contact_info(client_profile)
    # For currency, if not available in profile, we assume it should equal the account form's currency.
    profile_currency = client_profile.get("currency", "").strip() or acc_currency

    # --- Compare the fields ---
    if acc_name != profile_name or acc_passport != profile_passport:
        return False
    if acc_country != profile_country or acc_currency != profile_currency:
        return False
    if acc_email != profile_email or acc_address != profile_address:
        return False
    # For phone numbers, allow for formatting differences:
    if profile_phone not in acc_phone and acc_phone not in profile_phone:
        return False

    return True