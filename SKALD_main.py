# import statements
import subprocess
import json, os, requests, time, psutil
import logging
import yaml
import csv
import shutil
import tempfile
import numpy as np
import pandas as pd
import re
import glob
from SKALD.core import run_pipeline





class InlineListDumper(yaml.SafeDumper):
    pass

def represent_inline_list(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

# Extract numeric chunk number for correct sorting
def extract_chunk_number(path):
    match = re.search(r'chunk(\d+)', os.path.basename(path))
    return int(match.group(1)) if match else float('inf')



# Register custom representer for lists
InlineListDumper.add_representer(list, represent_inline_list)

def main_process(config):

    operations = config["operations"]
    dataset = config["data_type"]
    conf = config[dataset]

    if "chunking" in operations:
        try:
            suppress = conf.get("suppress", [])
            pseudonymize = conf.get("pseudonymize", [])
            encrypt = conf.get("encrypt", [])
            hardcoded_min_max = config.get("hardcoded_min_max", {})
            chunk_name = conf.get("chunk_name")
            number_of_chunks = conf.get("number_of_chunks")

            yaml_config = {
                "enable_k_anonymity": conf.get("enable_k_anonymity", True),
                "number_of_chunks": conf.get("number_of_chunks"),
                "chunk_name": chunk_name,
                "output_path": "generalized_chunk1.csv",
                "output_directory": "pipelineOutput",
                "key_directory": "keys",
                "log_file": "log.txt",
                "save_output": True,
                "suppress": suppress,
                "pseudonymize": pseudonymize,
                "encrypt": encrypt,
            }

            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_yaml:
                yaml.dump(
                    yaml_config,
                    tmp_yaml,
                    sort_keys=False,
                    Dumper=InlineListDumper
                )
                config_path = tmp_yaml.name

            yaml_debug_path = "pipelineOutput/last_used_config.yaml"
            os.makedirs("pipelineOutput", exist_ok=True)
            shutil.copyfile(config_path, yaml_debug_path)

            # --- Run SKALD pipeline ---
            result = run_pipeline(config_path=config_path)

            # If run_pipeline() returns only output file path
            if isinstance(result, str):
                concat_output = {
                    "output_file": result,
                    "status": "success",
                    "status_code": "0000",
                    "message": "Suppression/Pseudonymization/Encryption completed"
                }

            # If run_pipeline() returns a tuple
            elif isinstance(result, tuple) and len(result) == 5:
                final_rf, elapsed_time, lowest_dm_star, num_eq_classes, eq_class_stats = result
                concat_output = {
                    "result": final_rf,
                    "elapsed_time": elapsed_time,
                    "lowest_dm_star": int(lowest_dm_star),
                    "num_eq_classes": int(num_eq_classes),
                    "eq_class_stats": eq_class_stats,
                    "status": "success",
                    "status_code": "0000"
                }

            # Otherwise unknown output
            else:
                concat_output = {
                    "status": "failed",
                    "status_code": "1110",
                    "error_message": "Unexpected return format from run_pipeline()"
                }

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
