import requests
import json
import pandas as pd


SERVER_URL = "http://127.0.0.1:8071/process_SKALD"   # Or container IP if testing in TEE
CONFIG_PATH = "config_beneficiary.json"


with open(CONFIG_PATH, "r") as cfile:
        config = json.load(cfile)

response = requests.post(SERVER_URL, data=json.dumps(config))

print("\n=== Response Status ===")
print(response.status_code)
print("\n=== Response JSON ===")
try:
    print(json.dumps(response.json(), indent=2))
except Exception:
    print(response.text)
