import base64
import os
import zipfile
from os import PathLike

import requests

from jb_onboarding.preprocessing import Preprocessor

# Configuration - Replace these placeholders
API_KEY = "A8wwei-7ZHtA2TUFRW5AMRiwoFRijgAaIYO0AR6qeDk"
TEAM_NAME = "HANGUK ML"
BASE_URL = "https://hackathon-api.mlo.sehlat.io"


def start_game_session() -> tuple[str, str, str, dict]:
    # Configuration - Replace these placeholders

    """Start a new game session and retrieve session identifiers"""
    url = f"{BASE_URL}/game/start"
    headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}
    payload = {"player_name": TEAM_NAME}

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        # Extract values using correct keys from API response
        result = (
            data.get("session_id"),  # Corrected key name (underscore)
            data.get("client_id"),
            data.get("score", 0),  # New score field
            data.get("client_data", {}),
        )
    except requests.exceptions.HTTPError as e:
        if response.status_code == 422:
            print("Validation Error:", response.json())
        else:
            print(f"HTTP Error: {e}")
        result = None, None, None, None
    except requests.exceptions.RequestException as e:
        print(f"Request Failed: {e}")
        result = None, None, None, None

    return result


def save_erroneous_sample(account, description, passport, profile, client_idx, label):
    zip_filename = f"output/client_{client_idx}_{label}.zip"
    with zipfile.ZipFile(zip_filename, "w") as zf:
        zf.writestr("passport.png", passport)
        zf.writestr("account.pdf", account)
        zf.writestr("description.txt", description)
        zf.writestr("profile.docx", profile)
    print(f"Erroneous sample saved to {zip_filename}")


def get_decision(client_meta: dict):
    return "Accept"  # Placeholder for decision logic


def make_decision(session_id, client_id, decision):
    """Submit a game decision and return server response"""

    url = f"{BASE_URL}/game/decision"
    headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}
    payload = {"decision": decision, "session_id": session_id, "client_id": client_id}

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.HTTPError as e:
        # Check if the response object exists in the exception
        status_code = e.response.status_code if e.response else None
        print(f"Error making decision: {e}; Status code: {status_code}")
        return None, status_code


def main():
    os.makedirs("output", exist_ok=True)
    prep = Preprocessor()

    game_idx = 0
    client_idx = os.listdir("output")
    client_idx = [int(x.split("_")[1]) for x in client_idx if x.endswith(".zip")]
    client_idx = max(client_idx) + 1 if client_idx else 1

    while True:
        input("Press Enter to continue the game...")

        # Start the game session
        session_id, client_id, score, client_data = start_game_session()

        if session_id is None:
            print("Failed to start game session")
            continue

        while True:
            account = base64.b64decode(client_data.get("account"))
            description = base64.b64decode(client_data.get("description"))
            passport = base64.b64decode(client_data.get("passport"))
            profile = base64.b64decode(client_data.get("profile"))

            client_meta = prep(
                path_to_zip=None,
                passport=passport,
                description=description,
                account=account,
                profile=profile,
            )
            decision = get_decision(client_meta)
            result, err = make_decision(session_id, client_id, decision)

            if result is None:
                label = "Accept" if decision == "Reject" else "Reject"
                if err != 429:
                    save_erroneous_sample(account, description, passport, profile, client_idx, label)
                    client_idx += 1
                game_idx += 1
                break
            else:
                client_data = result.get("client_data")
                score = result.get("score")
            save_erroneous_sample(account, description, passport, profile, client_idx, decision)
            client_idx += 1
            print(f"Game {game_idx} - score {score}")


if __name__ == "__main__":
    main()
