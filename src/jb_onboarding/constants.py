DOCS = [
    "account.pdf",
    "description.txt",
    "passport.png",
    "profile.docx",
]


COUNTRIES = {
    "Austria": ["AUT", "Austrian"],
    "Belgium": ["BEL", "Belgian"],
    "Bulgaria": ["BGR", "Bulgarian"],
    "Croatia": ["HRV", "Croatian"],
    "Cyprus": ["CYP", "Cypriot"],
    "Czech Republic": ["CZE", "Czech"],
    "Denmark": ["DNK", "Danish"],
    "Estonia": ["EST", "Estonian"],
    "Finland": ["FIN", "Finnish"],
    "France": ["FRA", "French"],
    "Germany": ["DEU", "German"],
    "Greece": ["GRC", "Greek"],  # Eurostat uses EL instead of GR :cite[1]:cite[5]
    "Hungary": ["HUN", "Hungarian"],
    "Ireland": ["IRL", "Irish"],
    "Italy": ["ITA", "Italian"],
    "Latvia": ["LVA", "Latvian"],
    "Lithuania": ["LTU", "Lithuanian"],
    "Luxembourg": ["LUX", "Luxembourgish"],
    "Malta": ["MLT", "Maltese"],
    "Netherlands": ["NLD", "Dutch"],
    "Poland": ["POL", "Polish"],
    "Portugal": ["PRT", "Portuguese"],
    "Romania": ["ROU", "Romanian"],
    "Slovakia": ["SVK", "Slovak"],
    "Slovenia": ["SVN", "Slovenian"],
    "Spain": ["ESP", "Spanish"],
    "Sweden": ["SWE", "Swedish"],
}


default_rules = [
    "check_account_form",
    "check_age_consistency",
    # "check_education_background",
    "check_family_background_consistency",
    # "check_passport",
    # "check_occupation_history",
    "check_profile",
    "cross_check_account_form_client_profile",
    # "cross_check_account_form_passport",
    # "cross_check_passport_client_profile_form",
]
