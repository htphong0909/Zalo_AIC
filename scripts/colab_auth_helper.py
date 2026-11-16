#!/usr/bin/env python3
"""Helper script to manage Google Colab CLI OAuth authentication."""
from __future__ import annotations

import json
import os
import sys
from importlib import resources
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

PUBLIC_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/colaboratory",
    "https://www.googleapis.com/auth/drive.file",
]
REMOTE_REDIRECT_URI = "https://sdk.cloud.google.com/applicationdefaultauthcode.html"
TOKEN_PATH = Path(os.path.expanduser("~/.config/colab-cli/token.json"))
CACHE_PATH = Path(".cache/colab_pending_auth.json")


def generate_auth_url() -> str:
    config_resource = resources.files("colab_cli").joinpath("oauth_config.json")
    client_config = json.loads(config_resource.read_text(encoding="utf-8"))

    flow = InstalledAppFlow.from_client_config(client_config, PUBLIC_SCOPES)
    flow.redirect_uri = REMOTE_REDIRECT_URI
    auth_url, state = flow.authorization_url(prompt="consent", token_usage="remote", access_type="offline")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(
            {
                "client_config": client_config,
                "code_verifier": flow.code_verifier,
                "state": state,
                "auth_url": auth_url,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return auth_url


def complete_auth(auth_code: str) -> None:
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"Pending auth not found at {CACHE_PATH}. Run generate first.")

    data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    client_config = data["client_config"]
    code_verifier = data["code_verifier"]

    flow = InstalledAppFlow.from_client_config(client_config, PUBLIC_SCOPES)
    flow.redirect_uri = REMOTE_REDIRECT_URI
    flow.code_verifier = code_verifier

    clean_code = auth_code.strip()
    flow.fetch_token(code=clean_code)
    creds = flow.credentials

    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")
    print(f"Successfully saved credentials to {TOKEN_PATH}!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "complete":
        if len(sys.argv) < 3:
            print("Usage: python scripts/colab_auth_helper.py complete <AUTHORIZATION_CODE>")
            sys.exit(1)
        complete_auth(sys.argv[2])
    else:
        url = generate_auth_url()
        print("\n=== GOOGLE COLAB AUTHENTICATION URL ===")
        print(url)
