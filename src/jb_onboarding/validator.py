from jb_onboarding.checks.check_account_form import account_form_is_consistent
from jb_onboarding.checks.check_age_consistency import age_is_consistent
from jb_onboarding.checks.check_education_background import education_is_consistent
from jb_onboarding.checks.check_family_background_consistency import family_background_is_consistent

# from jb_onboarding.checks.check_occupation_history import
from jb_onboarding.checks.check_passport import passport_is_consistent
from jb_onboarding.checks.check_profile import profile_is_consistent
from jb_onboarding.checks.check_wealth_summary import wealth_is_consistent
from jb_onboarding.checks.cross_check_account_form_client_profile import account_form_and_client_profile_are_consistent
from jb_onboarding.checks.cross_check_account_form_passport import account_form_and_passport_are_consistent
from jb_onboarding.checks.cross_check_passport_client_profile_form import client_profile_and_passport_are_consistent


class ClientValidator:
    def __init__(self, rules: list):
        """_summary_
            [
            'check_account_form',
            'check_age_consistency',
            'check_education_background',
            'check_family_background_consistency'

            ]
        Args:
            rules (list): _description_
        """
        self.checkers = []
        for rule in rules:
            if rule == "check_account_form":
                self.checkers.append(account_form_is_consistent)
            elif rule == "check_age_consistency":
                self.checkers.append(age_is_consistent)
            elif rule == "check_education_background":
                self.checkers.append(education_is_consistent)
            elif rule == "check_family_background_consistency":
                self.checkers.append(family_background_is_consistent)
            elif rule == "check_passport":
                self.checkers.append(passport_is_consistent)
            elif rule == "check_wealth_summary":
                self.checkers.append(wealth_is_consistent)
            elif rule == "check_profile":
                self.checkers.append(profile_is_consistent)
            elif rule == "cross_check_account_form_client_profile":
                self.checkers.append(account_form_and_client_profile_are_consistent)
            elif rule == "cross_check_account_form_passport":
                self.checkers.append(account_form_and_passport_are_consistent)
            elif rule == "cross_check_passport_client_profile_form":
                self.checkers.append(client_profile_and_passport_are_consistent)
            else:
                raise ValueError(f"Unknown rule: {rule}")

    def validate(self, all_client_data: dict):
        for checker in self.checkers:
            if not checker(all_client_data):
                print(f"Validation failed for {checker.__name__}")
                return False

        return True
