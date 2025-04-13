import json
import re
from pathlib import Path


def education_is_consistent(data: dict):
    """
    Checks if education details in the description match those in the profile.

    """
    # --- Extract education data from profile ---
    profile = data.get("profile", {})
    description = data.get("description", {})

    try:
        higher = {
            "name": profile["higher_education"]["name"],
            "year": str(profile["higher_education"]["graduation_year"]),
        }
    except KeyError:
        # If secondary school data is missing, we consider the education inconsistent.
        return False

    # --- Process education text from description ---
    edu_text = description.get("Education Background", "").strip()
    if not edu_text:
        return False

    # Split the description into parts (each line could contain one education entry)
    parts = [part.strip() for part in edu_text.split("\n") if part.strip()]

    # --- Check that secondary school data appear in the description ---
    higher_found = any(higher["name"] in part and higher["year"] in part for part in parts)

    # The education is considered consistent if the secondary school is found and every higher edu record is found.
    return higher_found


# def education_is_consistent(data: dict):
#     """
#     Checks if education  details in the description match those in the profile.

#     New assumptions:
#       - The profile dictionary contains a single education record under:
#             profile["Account Holder – Personal Info"]["Education History"]
#         For example: "University of Turku (1989)"

#       - The description dictionary is expected to have an "Education Background" key
#         containing free-text education details, which should include both the institution name
#         and a 4-digit graduation year.

#     The function extracts the institution name and year from the profile, then verifies that
#     both are present (as substrings) somewhere in the education background text.
#     """
#     # --- Extract education info from profile ---
#     profile = data.get("profile", {})
#     description = data.get("description", {})
#     account_personal = profile.get("Account Holder – Personal Info", {})
#     edu_profile = account_personal.get("Education History", "").strip()
#     if not edu_profile:
#         return False

#     # Try to extract a 4-digit year from the education information.
#     year_match = re.search(r"(19|20)\d{2}", edu_profile)
#     year = year_match.group(0) if year_match else ""

#     # Assume the institution name is the part before an opening parenthesis (if present),
#     # otherwise use the whole string.
#     if "(" in edu_profile:
#         institution = edu_profile.split("(")[0].strip()
#     else:
#         institution = edu_profile

#     # --- Process education text from description ---
#     edu_text = description.get("Education Background", "").strip()
#     if not edu_text:
#         return False

#     # Check that the institution name and year appear in the education background text.
#     institution_found = institution in edu_text
#     year_found = year in edu_text

#     return institution_found and year_found
