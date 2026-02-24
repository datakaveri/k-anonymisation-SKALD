import re
import hashlib
import random
import string
import pandas as pd
import json
import os
import stat
import pyffx
import secrets
import hmac
import hashlib
import re

def derive_key(master_key: str, context: str) -> bytes:
    """
    Derive a per-segment key from the master key.
    This replaces FF1 tweaks for pyffx.
    """
    return hmac.new(
        master_key.encode(),
        context.encode(),
        hashlib.sha256
    ).digest()[:16]  # 128-bit key

def generate_fpe_key(path="fpe_key.json"):
    key = secrets.token_hex(16)  # 128-bit key
    with open(path, "w") as f:
        json.dump({"key": key}, f)
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    return key

def load_fpe_key(path="fpe_key.json"):
    with open(path, "r") as f:
        return json.load(f)["key"]


# -----------------------------
# Synthetic Data Generator
# -----------------------------

def generate_data(n=10):
    return pd.DataFrame({
        "id": [f"CUST-{10000+i}" for i in range(n)],
        "email": [f"user{i}@example.com" for i in range(n)],
        "pan": [f"{''.join(random.choices(string.ascii_uppercase, k=5))}"
                f"{random.randint(1000,9999)}"
                f"{random.choice(string.ascii_uppercase)}"
                for _ in range(n)]
    })


# -----------------------------
# 3. Pattern-Based Masking (Email)
# -----------------------------

def regex_mask(
    value: str,
    pattern: str,
    mask_groups: dict,
    mask_char: str = "*"
):
    """
    Generic regex-based masking.

    Parameters:
    - value: input string
    - pattern: regex with capturing groups
    - mask_groups: dict {group_index: mask_type}
        mask_type:
          - "full"      → replace entire group with mask_char
          - "partial"   → keep first char, mask rest
    - mask_char: masking character
    """
    if value is None:
        return None

    match = re.match(pattern, value)
    if not match:
        return value  # pattern does not match, return as-is

    groups = list(match.groups())

    for idx, mode in mask_groups.items():
        g = groups[idx - 1]  # group index is 1-based
        if g is None:
            continue

        if mode == "full":
            groups[idx - 1] = mask_char * len(g)
        elif mode == "partial":
            groups[idx - 1] = g[0] + mask_char * (len(g) - 1)

    # rebuild string
    masked = value
    for g, mg in zip(match.groups(), groups):
        masked = masked.replace(g, mg, 1)

    return masked


# -----------------------------
# 4. Character-Class Preserving Masking
# -----------------------------

def char_class_mask(value):
    def repl(c):
        if c.isdigit():
            return random.choice(string.digits)
        if c.isupper():
            return random.choice(string.ascii_uppercase)
        if c.islower():
            return random.choice(string.ascii_lowercase)
        return c
    return "".join(repl(c) for c in value)

# -----------------------------
# 4e. Class-Based Fixed Masking
# -----------------------------

def class_fixed_mask(value, letter="X", digit="0"):
    if value is None:
        return None
    out = []
    for c in value:
        if c.isdigit():
            out.append(digit)
        elif c.isalpha():
            out.append(letter)
        else:
            out.append(c)  # preserve separators
    return "".join(out)


# -----------------------------
# 6. Tokenization (Vault-Based)
# -----------------------------



class TokenVault:
    def __init__(self):
        self.forward = {}   # original -> token
        self.reverse = {}   # token -> original

    def tokenize(self, value):
        if value in self.forward:
            return self.forward[value]

        token = "TK-" + "".join(random.choices(string.digits, k=6))
        self.forward[value] = token
        self.reverse[token] = value
        return token

    def detokenize(self, token):
        return self.reverse.get(token)

    def save(self, path):
        """Persist vault to disk with restricted permissions."""
        data = {
            "forward": self.forward,
            "reverse": self.reverse
        }

        # Write file
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        # Restrict permissions: owner read/write only
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)

    @classmethod
    def load(cls, path):
        """Load vault from disk."""
        with open(path, "r") as f:
            data = json.load(f)

        vault = cls()
        vault.forward = data.get("forward", {})
        vault.reverse = data.get("reverse", {})
        return vault



# -----------------------------
# FPE Helpers (Segment-wise)
# -----------------------------
def get_ffx(derived_key: bytes):
    return pyffx.FFX(derived_key)
def fpe_pan_encrypt(pan, master_key):
    if pan is None:
        return None

    key_letters = derive_key(master_key, "pan_letters")
    key_digits  = derive_key(master_key, "pan_digits")
    key_suffix  = derive_key(master_key, "pan_suffix")

    cipher_letters_5 = pyffx.String(
        get_ffx(key_letters),
        alphabet=string.ascii_uppercase,
        length=5
    )
    cipher_digits_4 = pyffx.String(
        get_ffx(key_digits),
        alphabet=string.digits,
        length=4
    )
    cipher_letter_1 = pyffx.String(
        get_ffx(key_suffix),
        alphabet=string.ascii_uppercase,
        length=1
    )

    return (
        cipher_letters_5.encrypt(pan[:5]) +
        cipher_digits_4.encrypt(pan[5:9]) +
        cipher_letter_1.encrypt(pan[9])
    )
def fpe_pan_decrypt(pan_enc, master_key):
    key_letters = derive_key(master_key, "pan_letters")
    key_digits  = derive_key(master_key, "pan_digits")
    key_suffix  = derive_key(master_key, "pan_suffix")

    cipher_letters_5 = pyffx.String(
        get_ffx(key_letters),
        alphabet=string.ascii_uppercase,
        length=5
    )
    cipher_digits_4 = pyffx.String(
        get_ffx(key_digits),
        alphabet=string.digits,
        length=4
    )
    cipher_letter_1 = pyffx.String(
        get_ffx(key_suffix),
        alphabet=string.ascii_uppercase,
        length=1
    )

    return (
        cipher_letters_5.decrypt(pan_enc[:5]) +
        cipher_digits_4.decrypt(pan_enc[5:9]) +
        cipher_letter_1.decrypt(pan_enc[9])
    )




if __name__ == "__main__":
    df = generate_data(5)
    vault = TokenVault()

    # --- FPE key management ---
    if not os.path.exists("fpe_key.json"):
        fpe_key = generate_fpe_key("fpe_key.json")
    else:
        fpe_key = load_fpe_key("fpe_key.json")

    # --- Existing transforms ---
    df["email_pattern_mask"] = df["email"].apply(
        lambda x: regex_mask(x, r"([^@]+)@(.+)", {1: "partial"})
    )

    df["pan_char_class"] = df["pan"].apply(char_class_mask)
    df["pan_fixed_mask"] = df["pan"].apply(class_fixed_mask)
    # --- FPE ---
    df["pan_fpe"] = df["pan"].apply(lambda x: fpe_pan_encrypt(x, fpe_key))

    # --- Tokenization ---
    df["id_tokenized"] = df["id"].apply(vault.tokenize)
    vault.save("vault.json")

    df.to_csv("processed.csv", index=False)

