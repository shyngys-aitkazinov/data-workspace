import json
import re
from pathlib import Path


def wealth_is_consistent(description, profile):
    """
    Checks if the 'Wealth Summary' in the description contains three separate sentences:
      - One sentence containing the savings value from profile["aum"]["savings"].
      - One sentence containing the inheritance value from profile["aum"]["inheritance"] and all details 
        from profile["inheritance_details"].
      - One sentence containing all details for each real estate asset in profile["real_estate_details"].
    
    If a particular group is not available in the profile, that group is skipped.
    If no groups are required, the function returns True.
    The groups that are found must be in three distinct sentences.
    """
    try:
        wealth_text = description["Wealth Summary"]
        # Split by newline characters and remove empty lines.
        sentences = [s.strip() for s in wealth_text.splitlines() if s.strip()]
    except KeyError:
        return False

    required_groups = {}  # Will hold group_name -> sentence index

    # ---- Group 1: Savings ----
    aum = profile.get("aum", {})
    savings = str(aum.get("savings", "")).strip()
    if savings:   # Only require if provided.
        savings_idx = None
        for i, sentence in enumerate(sentences):
            # Use re.escape to avoid regex interpretation of any special characters.
            if re.search(r'\b' + re.escape(savings) + r'\b', sentence):
                savings_idx = i
                break
        if savings_idx is None:
            return False
        required_groups["savings"] = savings_idx

    # ---- Group 2: Inheritance and its details ----
    inheritance = str(aum.get("inheritance", "")).strip()
    inheritance_details = profile.get("inheritance_details", {})
    # Require this group only if both an inheritance value and details are provided.
    if inheritance and inheritance_details:
        in_relationship = str(inheritance_details.get("relationship", "")).strip()
        in_year = str(inheritance_details.get("inheritance year", "")).strip()
        in_profession = str(inheritance_details.get("profession", "")).strip()
        inheritance_idx = None
        for i, sentence in enumerate(sentences):
            if ( re.search(r'\b' + re.escape(inheritance) + r'\b', sentence)
                 and re.search(r'\b' + re.escape(in_year) + r'\b', sentence)
                 and re.search(r'\b' + re.escape(in_relationship) + r'\b', sentence)
                 and re.search(r'\b' + re.escape(in_profession) + r'\b', sentence) ):
                inheritance_idx = i
                break
        if inheritance_idx is None:
            return False
        required_groups["inheritance"] = inheritance_idx

    # ---- Group 3: Real estate details ----
    real_estate_details = profile.get("real_estate_details", [])
    if real_estate_details:
        # Assume real_estate_details is a list of strings.
        real_estate_idx = None
        for i, sentence in enumerate(sentences):
            all_found = True
            for asset in real_estate_details:
                asset_detail = str(asset).strip()
                if asset_detail and not re.search(r'\b' + re.escape(asset_detail) + r'\b', sentence):
                    all_found = False
                    break
            if all_found:
                real_estate_idx = i
                break
        if real_estate_idx is None:
            return False
        required_groups["real_estate"] = real_estate_idx

    # If no groups were required (i.e. none of the three pieces of wealth info were provided),
    # return True.
    if not required_groups:
        return True

    # Ensure that the sentences identified for the required groups are all different.
    indices = list(required_groups.values())
    if len(indices) != len(set(indices)):
        return False

    return True
