"""
At-rest encryption for skeleton tensors (the only patient-derived data this
pipeline persists to disk -- raw video is deleted after extraction).

IMPORTANT -- this is a development-grade key management scheme: a single
Fernet key stored in a local file. It is enough to ensure skeleton files are
not plaintext-readable if the disk/backup is copied off this machine, but it
is NOT a substitute for real key management. Before handling real patient
data in production, replace SKELETON_KEY_PATH with a managed KMS (AWS KMS,
GCP KMS, Azure Key Vault, HashiCorp Vault) and per-tenant/per-record key
rotation. Do not commit the key file to git (it's gitignored below).
"""
import os
from cryptography.fernet import Fernet
import numpy as np
import io

SKELETON_KEY_PATH = os.environ.get("SKELETON_KEY_PATH", os.path.join(".keys", "skeleton.key"))


def generate_key(path=SKELETON_KEY_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        raise FileExistsError(f"Key already exists at {path}. Refusing to overwrite.")
    key = Fernet.generate_key()
    with open(path, "wb") as f:
        f.write(key)
    os.chmod(path, 0o600)
    print(f"Generated new encryption key at {path}. Back this up securely -- "
          f"losing it makes all encrypted skeleton data permanently unreadable.")
    return key


def load_key(path=SKELETON_KEY_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No encryption key found at {path}. Run `python crypto_utils.py init` first."
        )
    with open(path, "rb") as f:
        return f.read()


def get_fernet():
    return Fernet(load_key())


def save_encrypted_npy(array, path):
    """Serialize a numpy array and write it encrypted to `path`."""
    buf = io.BytesIO()
    np.save(buf, array)
    token = get_fernet().encrypt(buf.getvalue())
    with open(path, "wb") as f:
        f.write(token)


def load_encrypted_npy(path):
    with open(path, "rb") as f:
        token = f.read()
    plaintext = get_fernet().decrypt(token)
    return np.load(io.BytesIO(plaintext))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "init":
        generate_key()
    else:
        print("Usage: python crypto_utils.py init")
