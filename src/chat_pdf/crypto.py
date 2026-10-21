import base64
import hashlib

from cryptography.fernet import Fernet


class Crypto:
    """Encrypts provider API keys at rest. APP_SECRET_KEY may be any string."""

    def __init__(self, secret: str):
        key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())
        self._fernet = Fernet(key)

    def encrypt(self, value: str) -> bytes:
        return self._fernet.encrypt(value.encode())

    def decrypt(self, token: bytes) -> str:
        return self._fernet.decrypt(token).decode()


def mask(key: str | None) -> str | None:
    """`sk-proj-abcd1234` -> `sk-****1234`; short keys are fully hidden."""
    if not key:
        return None
    if len(key) <= 8:
        return "****"
    prefix = "sk-" if key.startswith("sk-") else ""
    return f"{prefix}****{key[-4:]}"
