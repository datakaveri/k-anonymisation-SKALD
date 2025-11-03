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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_yaml:
            yaml.dump(
                config,
                tmp_yaml,
                sort_keys=False,
                Dumper=InlineListDumper
            )
            config_path = tmp_yaml.name

        yaml_debug_path = "pipelineOutput/last_used_config.yaml"
        os.makedirs("pipelineOutput", exist_ok=True)
        shutil.copyfile(config_path, yaml_debug_path)

        # Run SKALD pipeline
        final_rf, elapsed_time, lowest_dm_star, num_eq_classes, eq_class_stats = run_pipeline(config_path=config_path)

        concat_output = {
            "result": final_rf,
            "elapsed_time": elapsed_time,
            "lowest_dm_star": int(lowest_dm_star),
            "num_eq_classes": int(num_eq_classes),
            "eq_class_stats": eq_class_stats,
            "status": "success",
            "status_code": "0000"
        }

        return concat_output
