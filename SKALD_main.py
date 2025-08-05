# import statements
import subprocess
import json, os, requests
import logging
import yaml
import csv
import shutil
import tempfile
import numpy as np
import pandas as pd
import os
import re
import glob
from SKALD.core import run_pipeline
from utils_arx import average_best_node, calculate_DM, convert_ui_config_to_arx


import sys

class InlineListDumper(yaml.SafeDumper):
    pass

def represent_inline_list(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

def extract_chunk_number(path):
    match = re.search(r'(\d+)', os.path.basename(path))
    return int(match.group(1)) if match else float('inf')
# Register custom representer for lists
InlineListDumper.add_representer(list, represent_inline_list)

def main_process(config):
        operations = config["operations"]
        dataset = config["data_type"]
        conf = config[dataset]

        if "chunking" in operations:
            try:
                categorical = conf["quasi_identifiers"].get("categorical", [])
                numerical = conf["quasi_identifiers"].get("numerical", [])
                suppress = conf.get("suppress", [])
                pseudonymize = conf.get("pseudonymize", [])
                hardcoded_min_max = config.get("hardcoded_min_max", {})
                chunk_directory = conf.get("chunk_directory", "5K_rows_data")
                number_of_chunks = conf.get("number_of_chunks")
                sizes = {
                    qid["column"]: qid["size"]
                    for qid in numerical
                    if "size" in qid
                }


                for qid in numerical:
                    qid.pop("size", None)
                
                yaml_config = {
                    "number_of_chunks": conf.get("number_of_chunks"),
                    "k": conf["k_anonymize"]["k"],
                    "l": conf["l_diversity"]["l"],
                    "max_number_of_eq_classes": 62500,
                    "suppression_limit": conf["suppression_limit"],
                    "chunk_directory": "5K_rows_data",
                    "output_path": "generalized_chunk1.csv",
                    "log_file": "log.txt",
                    "save_output": True,
                    "suppress": suppress,
                    "pseudonymize": pseudonymize,
                    "quasi_identifiers": {
                        "numerical": numerical,
                        "categorical": categorical
                    },
                    "sensitive_parameter":"Health Condition",
                    "bin_width_multiplication_factor": sizes,
                    "hardcoded_min_max": hardcoded_min_max
                }



                with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_yaml:
                    yaml.dump(
                        yaml_config,
                        tmp_yaml,
                        sort_keys=False,
                        Dumper=InlineListDumper
                    )
                    config_path = tmp_yaml.name

                # 💾 Save a copy of the generated YAML
                yaml_debug_path = "pipelineOutput/last_used_config.yaml"
                os.makedirs("pipelineOutput", exist_ok=True)
                shutil.copyfile(config_path, yaml_debug_path)


                output_file = config.get("output_path", "generalized_chunk1.csv")
                if not os.path.isfile(output_file):
                    raise FileNotFoundError(f"Expected output file not found: {output_file}")
                with open(output_file, "r") as f:
                    reader = csv.DictReader(f)
                    anonymized_output = list(reader)

                final_rf, elapsed_time, lowest_dm_star, num_eq_classes, eq_class_stats = run_pipeline(config_path=config_path)

                #---------------------------------------------------------------------------------


                dataset_type = dataset
                arx_ready_config = convert_ui_config_to_arx(config)
                k_anon_config = arx_ready_config["k_anonymize"]
                arx_ready_config["k"] = k_anon_config["k"] 
                url = 'http://localhost:8070/api/arx/process'
                headers = {'Content-Type': 'application/json'}
                #fileList = sorted(glob.glob(os.path.join(chunk_directory, "*.csv")))
                #print(fileList)
                #print(config["sensitive_parameter"])
                fileList = sorted(glob.glob(os.path.join(chunk_directory, "*.csv")), key=extract_chunk_number)
                saved_first_chunk = False
                first_chunk_df = pd.read_csv(fileList[0])

                for chunk_path in fileList[:number_of_chunks]:
                    abs_chunk_path = os.path.abspath(chunk_path)
                    print(f"[ARX] Sending chunk: {chunk_path}")

                    #print("ARX config:", json.dumps(arx_ready_config, indent=2)) #debug statement for config file

                    k_anon_params = {
                        "dataset_name": abs_chunk_path,
                        "datasetType": dataset_type,
                        "num_chunks": arx_ready_config.get("number_of_chunks", len(fileList)),
                        "k": k_anon_config["k"],
                        "l": k_anon_config["l"],
                        "suppress_columns": ','.join(arx_ready_config.get('suppress', [])),
                        "pseudonymize_columns": ','.join(arx_ready_config.get('pseudonymize', [])),
                        "generalized_columns": ','.join(arx_ready_config.get('generalize', [])),
                        #"insensitive_columns": ','.join(arx_ready_config.get('insensitive_columns', [])),
                        "insensitive_columns": "id",
                        "sensitive_column": "Health Condition",  # Use the sensitive parameter from the config
                        "widths": arx_ready_config["width"],
                        "num_levels": arx_ready_config["levels"],
                        "suppression_limit": arx_ready_config["suppression_limit"]
                    }
                    #print(k_anon_params)
                    #response = requests.get(url, params=k_anon_params)
                    response = requests.post(url, headers=headers, data=json.dumps(k_anon_params))
                    concat_output = json.loads(response.text)

                    if response.status_code == 200:
                        print(f"[ARX] Success for chunk: {chunk_path}")
                        result = response.json()

                        if not saved_first_chunk:
                            output_path = f"pipelineOutput/arx_output_{os.path.basename(chunk_path)}.json"
                            with open(output_path, "w") as f:
                                json.dump(result, f, indent=2)
                            saved_first_chunk = True
                    else:
                        print(f"[ARX] Failed for chunk: {chunk_path}")
                        print("Status:", response.status_code)
                        print("Response:", response.text)
                        raise RuntimeError("ARX request failed.")
                    
                print("\n[ARX] All chunks processed. Now calculating average best node...")
                #print(fileList)
                base_chunk_path = fileList[0].replace("1.csv", "{i}.csv")
                k_anon_params.pop("dataset_name", None)
                best_node = average_best_node(arx_ready_config=arx_ready_config,num_chunks=number_of_chunks,url=url,headers=headers,base_chunk_path=base_chunk_path,params_template=k_anon_params)

                print(f"[ARX] Final averaged node across chunks: {best_node}")

                dm_star_value,arx_eq_classes = calculate_DM(arx_ready_config,number_of_chunks,fileList[0], best_node)
                print(f"[ARX] DM* score for average node: {dm_star_value}")



                #------------------------------------------------------------------------------------------

                summary_output = {
                    "anonymised_output": anonymized_output,
                    "result": final_rf,
                    "elapsed_time": elapsed_time,
                    "lowest_dm_star": int(lowest_dm_star),
                    "arx_dm_star": int(dm_star_value),
                    "num_eq_classes": int(num_eq_classes),
                    "arx_eq_classes": int(arx_eq_classes),
                    "eq_class_stats": eq_class_stats,
                    "arx_output_json": result,  # <-- Add ARX JSON here
                    "first_chunk_data": first_chunk_df.to_dict(orient="records"),  # <-- Add First Chunk Data
                    "status": "success",
                    "status_code": "0000"
                }




                concat_output = summary_output



            except subprocess.CalledProcessError as e:
                concat_output = {
                    "status": "failed",
                    "status_code": "1111",
                    "error_message": "SKALD execution failed",
                    "stdout": e.stdout,
                    "stderr": e.stderr
                }
            except Exception as e:
                concat_output = {
                    "status": "failed",
                    "status_code": "1112",
                    "error_message": str(e)
                }

            return concat_output