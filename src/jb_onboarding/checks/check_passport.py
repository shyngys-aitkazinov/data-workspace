
import json
from datetime import datetime


def passport_is_consistent(passport):
    """
    Checks consistency on the passport file only
    """

    # Reading data
    first_name = passport["first_name"]
    middle_name = passport["middle_name"]
    last_name = passport["last_name"]
    country = passport["country"]
    country_code = passport["country_code"]
    nationality = passport["nationality"]
    number = passport["passport_number"]
    birth_date_str = passport["birth_date"]
    gender = passport["gender"]
    mrz_line_0 = passport["passport_mrz"][0]
    mrz_line_1 = passport["passport_mrz"][1]
    issue_date_str = passport["passport_issue_date"]
    expiry_date_str = passport["passport_expiry_date"]

    if gender == "":
        return False

    expected_mrz_line0 = "P<" + country_code + last_name.upper() + "<<" + first_name.upper()
    if middle_name != "":
        expected_mrz_line0 += "<" + middle_name.upper()
    while len(expected_mrz_line0) < len(mrz_line_0):
        expected_mrz_line0 += "<"
        
    expected_mrz_line1 = number + country_code + birth_date_str[2:4] + birth_date_str[5:7] + birth_date_str[8:10]
    while len(expected_mrz_line1) < len(mrz_line_1):
        expected_mrz_line1 += "<"

    # Check consistency of mrz 
    if mrz_line_0 != expected_mrz_line0 or mrz_line_1 != expected_mrz_line1:
        return False
    
    ## DATES CHECKS
    birth_date = datetime.strptime(birth_date_str, "%Y-%m-%d").date()
    issue_date = datetime.strptime(issue_date_str, "%Y-%m-%d").date()
    expiry_date = datetime.strptime(expiry_date_str, "%Y-%m-%d").date()
    current_date = datetime.strptime("2025-04-01", "%Y-%m-%d").date()

    if birth_date > issue_date or issue_date > expiry_date or birth_date > current_date or issue_date > current_date:
        return False

    with open('country_mappings.json', 'r') as f:
        country_data = json.load(f)     
    assert country in country_data

    if country_data[country][0] != country_code or country_data[country][1] != nationality:
        return False
    return True
