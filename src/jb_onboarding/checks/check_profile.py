import re


def profile_is_consistent(data: dict) -> bool:
    """
    Checks consistency of education, employment, assets, and investment profile
    using the new structured format.
    """
    profile = data.get("profile", {})
    try:
        # Extract birth year from ISO date
        birth_year = int(profile["birth_date"].split("-")[0])
    except (KeyError, ValueError):
        return False

    # Higher education validation
    higher_ed = profile.get("higher_education", [])
    if isinstance(higher_ed, dict):  # Handle single-entry format
        higher_ed = [higher_ed]

    for edu in higher_ed:
        try:
            grad_year = edu["graduation_year"]
            edu_type = edu.get("type", "tertiary").lower()

            # Minimum age rules
            if edu_type == "secondary":
                if not (16 <= (grad_year - birth_year) <= 25):
                    return False
            elif edu_type == "tertiary":
                if grad_year - birth_year < 18:
                    return False
        except KeyError:
            continue

    # Employment history validation
    employment = profile.get("employment_background", {})
    if employment:
        try:
            start_year = employment["since"]
            status = employment["status"].lower()

            # Validate employment start age
            if start_year - birth_year < 16:
                return False

            # Validate retirement if applicable
            if status == "retired":
                if start_year <= grad_year:
                    return False
        except KeyError:
            pass

    # Investment profile validation
    valid_risk = ["low", "moderate", "considerable", "high"]
    valid_horizon = ["short", "medium", "long-term"]
    valid_mandate = ["advisory", "discretionary"]

    if (
        profile["investment_risk_profile"].lower() not in valid_risk
        or profile["investment_horizon"].lower() not in valid_horizon
        or profile["type_of_mandate"].lower() not in valid_mandate
    ):
        return False

    return True
