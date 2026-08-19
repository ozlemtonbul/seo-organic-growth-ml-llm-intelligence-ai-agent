from __future__ import annotations

from pathlib import Path
from typing import Sequence

from google.oauth2 import service_account
from google.oauth2.service_account import Credentials


def get_google_credentials(
    service_account_file: str,
    scopes: Sequence[str],
) -> Credentials:
    """
    Load Google service-account credentials from a JSON file.

    Parameters
    ----------
    service_account_file:
        Path to the Google service-account JSON file.
    scopes:
        Google API permission scopes required by the client.

    Returns
    -------
    Credentials
        Authenticated Google service-account credentials.
    """
    if not service_account_file:
        raise ValueError(
            "A Google service-account file path is required."
        )

    credentials_path = Path(service_account_file)

    if not credentials_path.exists():
        raise FileNotFoundError(
            "Google service-account file was not found: "
            f"{credentials_path.resolve()}"
        )

    if not credentials_path.is_file():
        raise ValueError(
            "Google service-account path must point to a file."
        )

    if credentials_path.suffix.lower() != ".json":
        raise ValueError(
            "Google service-account credentials must be stored "
            "in a JSON file."
        )

    return service_account.Credentials.from_service_account_file(
        str(credentials_path),
        scopes=list(scopes),
    )