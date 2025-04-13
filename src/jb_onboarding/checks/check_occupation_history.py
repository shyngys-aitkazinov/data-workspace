import json
import re
from pathlib import Path


def employment_is_consistent(data):
    """
    Verify if employment details in description match the profile's current employment data.
    Checks for:
    1. Company name mention in occupation history
    2. Presence of start year from profile
    3. No conflicting dates for current employment status
    """
    profile = data.get("profile", {})
    description = data.get("description", {})
    try:
        employment = profile["employment_background"]
        occupation_text = description.get("Occupation History", "")
        sentences = [s.strip() for s in occupation_text.split("\n") if s.strip()]
    except KeyError:
        return False

    # Filter out summary sentences containing experience duration
    filtered_sentences = [s for s in sentences if "years of experience" not in s.lower()]

    # Extract key employment data
    target_company = employment.get("company", "").lower().strip()
    start_year = str(employment.get("since", ""))
    current_status = employment.get("status", "")
    previous_occupation = employment.get("previous_profession", "").lower().strip()

    for sentence in filtered_sentences:
        sentence_lower = sentence.lower()

        # 1. Company match check
        if target_company not in sentence_lower:
            continue

        if previous_occupation not in sentence_lower:
            continue

        # 2. Year validation
        found_years = re.findall(r"\b\d{4}\b", sentence)
        if not found_years:
            continue

        # For current employment statuses, should only have start year
        if current_status in ["employee", "self-employed"]:
            if start_year not in found_years or len(found_years) > 1:
                continue

        # For past statuses (retired etc), start year should exist
        else:
            if start_year not in found_years:
                continue

        return True

    return False
