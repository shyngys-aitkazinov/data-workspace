from jb_onboarding.checks.check_account_form import account_form_is_consistent
from jb_onboarding.checks.check_age_consistency import age_is_consistent
from jb_onboarding.checks.check_education_background import education_is_consistent
from jb_onboarding.checks.check_family_background_consistency import family_background_is_consistent
from jb_onboarding.checks.check_occupation_history import employment_is_consistent
from jb_onboarding.checks.check_passport_consistency import passport_is_consistent
from jb_onboarding.checks.check_profile import profile_is_consistent
from jb_onboarding.checks.check_wealth_summary import wealth_is_consistent
from jb_onboarding.checks.cross_check_account_form_client_profile import account_form_and_client_profile_are_consistent
from jb_onboarding.checks.cross_check_account_form_passport import account_form_and_passport_are_consistent
from jb_onboarding.checks.cross_check_passport_client_profile_form import client_profile_and_passport_are_consistent


class ClientValidator:
    def __init__(self, rules: list, sign_model_id: str | None = None):
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
                continue
                # self.checkers.append(wealth_is_consistent)
            elif rule == "check_occupation_history":
                self.checkers.append(employment_is_consistent)
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

    def __call__(self, all_client_data: dict, flag: bool = True) -> bool:
        """

        Args:
            all_client_data (dict): client data dictionary containing all the necessary information
            flag (bool, optional): to skip related to passport checkers Defaults to True.

        Returns:
            bool: Valid or not
        """
        for checker in self.checkers:
            print(checker.__name__)
            print(checker(all_client_data))
            if flag and "passport" in checker.__name__:
                continue
            if not checker(all_client_data):
                return False

        return True
