from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import json
import configparser
import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SKALD_main import main_process 

app = Flask(__name__)
CORS(app, origins="*", allow_headers="*")
server_config = configparser.ConfigParser()
server_config.read('server_config.cfg')

main_server_ip = server_config.get('SKALD_SERVER', 'ip')
main_server_port = server_config.get('SKALD_SERVER', 'port')

@app.route("/test_SKALD", methods=["GET"])
def test_server():
    return jsonify({"message": "SKALD server running!"})

@app.route("/process_SKALD", methods=["POST"])
def process_k_anon_chunk():
    try:
        config = request.get_json(force=True)
        if not config:
            raise ValueError("Empty or invalid JSON received")

 
        result = main_process(config)

        os.makedirs("pipelineOutput", exist_ok=True)
        with open("pipelineOutput/SKALD_output.json", "w") as f_out:
            json.dump(result, f_out, indent=2)
        print("SKALD response saved successfully")

        return jsonify(result), 200 if result.get("status") == "success" else 500

    except Exception as e:
        response = {
            "status": "failed",
            "status_code": "9999",
            "error_message": str(e)
        }
        return jsonify(response), 500

@app.route("/get_dataset_names", methods=['GET'])
def get_dataset_names():
    url = server_config.get('RESOURCE_SERVER', 'url')+"uploads"
    username=server_config.get('RESOURCE_SERVER', 'username')
    password=server_config.get('RESOURCE_SERVER', 'password')
    headers = {
        "Content-Type": "application/json"
    }

    response = requests.get(url, headers=headers, auth=(username, password))
    data = response.json()
    filenames = data.get('files', [])

    if response.status_code == 200:
        print('Fetched dataset names successfully.')
    else:
        print('Failed to fetch dataset names. Status code:', response.status_code)
    return jsonify(filenames), 200



if __name__ == "__main__":
    #app.run(host=main_server_ip, port=int(main_server_port), debug=True)
    app.run(host='0.0.0.0',debug=True,port = 8071)