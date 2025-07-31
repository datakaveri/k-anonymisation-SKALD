# import statements
import subprocess
import json, os, requests
import logging
import yaml
import csv
import shutil
import tempfile
from SKALD.core import run_pipeline


import sys

class InlineListDumper(yaml.SafeDumper):
    pass

def represent_inline_list(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

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
                hardcoded_min_max = config.get("hardcoded_min_max", {})
                sizes = {
                    qid["column"]: qid["size"]
                    for qid in numerical
                    if "size" in qid
                }


                for qid in numerical:
                    qid.pop("size", None)

                yaml_config = {
                    "number_of_chunks": conf.get("number_of_chunks", 1),
                    "k": conf["k_anonymize"]["k"],
                    "l": conf["l_diversity"]["l"],
                    "max_number_of_eq_classes": 12500,
                    "suppression_limit": conf["suppression_limit"],
                    "chunk_directory": "small_datachunks",
                    "output_path": "generalized_chunk1.csv",
                    "log_file": "log.txt",
                    "save_output": True,
                    "quasi_identifiers": {
                        "numerical": numerical,
                        "categorical": categorical
                    },
                    "sensitive_parameter": conf.get("sensitive_parameter", "Health Condition"),
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

                final_rf, elapsed_time = run_pipeline(config_path=config_path)

                summary_output = {
                    "anonymised_output": anonymized_output,
                    "result": final_rf,
                    "elapsed_time": elapsed_time,
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