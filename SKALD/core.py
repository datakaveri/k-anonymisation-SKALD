import os
import time
import logging
import psutil
import pandas as pd
import yaml
import tempfile
import json

# SKALD modules
from SKALD.quasi_identifier import QuasiIdentifier
from SKALD.generalization_ri import OLA_1
from SKALD.generalization_rf import OLA_2
from SKALD.utils import format_time, log_performance
from SKALD.config_validation import load_config
from SKALD.encoder import encode_numerical_columns
from SKALD.chunk_processinng import process_chunks_for_histograms
from SKALD.generalize_chunk import generalize_first_chunk   
from SKALD.build_QI import build_quasi_identifiers
from SKALD.chunking import split_csv_by_ram

class InlineListDumper(yaml.SafeDumper):
    pass

def represent_inline_list(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

def run_pipeline(config_path="config.yaml", k=None, chunks=None, chunk_dir=None):
    """
    Main SKALD pipeline: loads config, encodes QIs, builds OLA trees,
    processes histograms, calculates final bin widths, and optionally generalizes the first chunk.

    Args:
        config_path (str): Path to the configuration file.
        k (int, optional): Override k-anonymity parameter.
        chunks (int, optional): Override number of chunks to process.
        chunk_dir (str, optional): Directory containing chunk CSV files.

    Returns:
        tuple: (final_rf, elapsed_time, lowest_dm_star, num_eq_classes, eq_class_stats)
    """

    # === LOAD CONFIGURATION ===
    config = load_config(config_path)

    # Handle suppression, pseudonymization, encryption only mode
    print("Running suppression + pseudonymization + encryption ")

    from SKALD.preprocess import suppress, pseudonymize, encrypt_columns

    os.makedirs(config.output_directory, exist_ok=True)
    os.makedirs(config.key_directory, exist_ok=True)
    split_csv_by_ram("data")
    chunks = os.listdir("chunks")

    for chunk in chunks:
        chunk_path = os.path.join("chunks", chunk)
        df = pd.read_csv(chunk_path)

        if config.suppress:
            print(f"Suppressing columns: {config.suppress}")
            df = suppress(df, config.suppress)

        if config.pseudonymize:
            print(f"Pseudonymizing columns: {config.pseudonymize}")
            df = pseudonymize(df, config.pseudonymize)

        if config.encrypt:
            print(f"Encrypting columns: {config.encrypt}")
            df = encrypt_columns(df, config.encrypt)

        temp_path = chunk_path + ".tmp"
        df.to_csv(temp_path, index=False)
        os.replace(temp_path, chunk_path) 

        
        if not config.enable_k_anonymity :
            json_output = df.to_json(orient="records")
            output_json_dir = os.path.join(os.getcwd(), "output")
            os.makedirs(output_json_dir, exist_ok=True)
            output_json_path = os.path.join(output_json_dir, f"{os.path.basename(chunk)}_preprocessed.json")

            with open(output_json_path, "w") as f:
                f.write(json_output)

            print(f"JSON output saved at: {output_json_path}")
        print("\nCompleted suppression/pseudonymization/encryption phase.")

    
    if config.enable_k_anonymity:
        # Setup logging
        log_file = "log.txt"
        if not os.path.exists(log_file):
            with open(log_file, "w") as f:
                f.write("=== SKALD Log File Created ===\n")

        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        logger = logging.getLogger("SKALD")
        
        if k is not None:
            config.k = k


        # === PIPELINE PARAMETERS ===
        k = config.k
        l = config.l
        chunk_dir = "chunks"
        output_path = config.output_path
        suppression_limit = config.suppression_limit

        # Estimate memory-based limit for OLA tree
        process = psutil.Process(os.getpid())
        available_ram_bytes = psutil.virtual_memory().available 
        max_equivalence_classes = available_ram_bytes // 10000
        logger.info(f"Available RAM: {available_ram_bytes}, maximum number of equivalence classes: {max_equivalence_classes}")

        # Columns and settings from config
        sensitive_parameter = config.sensitive_parameter
        categorical_columns = [cat_qi.column for cat_qi in config.quasi_identifiers.categorical]
        numerical_columns_info = [
            {"column": num_qi.column, "encode": num_qi.encode, "type": num_qi.type}
            for num_qi in config.quasi_identifiers.numerical
        ]
        hardcoded_min_max = config.hardcoded_min_max
        multiplication_factors = config.bin_width_multiplication_factor
        print("multiplication_factors:", multiplication_factors)
        # === VALIDATE CHUNK DIRECTORY ===
        if not os.path.exists(chunk_dir):
            raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")

        all_files = sorted([f for f in os.listdir(chunk_dir) if f.endswith(".csv")])
        if not all_files:
            raise ValueError("No CSV files found in the chunk directory.")
        chunk_files = all_files
        n = len(chunk_files)
        print(f"Found {n} chunk files for processing.")
        start_time = time.time()

        # === ENCODE NUMERICAL COLUMNS ACROSS ALL CHUNKS ===
        encoding_maps = encode_numerical_columns(
            chunk_files, chunk_dir, numerical_columns_info
        )

        # === DEFINE QUASI-IDENTIFIERS ===
        # Returns: list of QuasiIdentifier objects and list of column names
        quasi_identifiers, all_quasi_columns = build_quasi_identifiers(
            numerical_columns_info,
            categorical_columns,
            encoding_maps,
            hardcoded_min_max
        )
        print("Selected quasi-identifiers:", all_quasi_columns)

        # === CALCULATE TOTAL RECORDS ===
        total_records = 0
        if chunk_files:
            first_chunk = pd.read_csv(os.path.join(chunk_dir, chunk_files[0]))
            total_records = len(first_chunk) * n

        # === BUILD INITIAL OLA_1 TREE FOR Ri ===
        print("\nBuilding initial tree and finding Ri values...")
        ola_1 = OLA_1(quasi_identifiers, n, max_equivalence_classes, multiplication_factors)
        ola_1.build_tree()
        ola_1.find_smallest_passing_ri(n)
        initial_ri = ola_1.get_optimal_ri()
        print("Initial bin widths (Ri):", initial_ri)
        logger.info(f"Initial bin widths (Ri): {initial_ri}")
        log_performance(logger,"OLA_1 tree", start_time)

        # === BUILD OLA_2 TREE FOR FINAL BIN WIDTHS ===
        ola_2 = OLA_2(quasi_identifiers, total_records, suppression_limit, multiplication_factors, sensitive_parameter)
        print("\nBuilding second tree with initial Ri values as root...")
        ola_2.build_tree(initial_ri)

        # === PROCESS CHUNKS TO COLLECT HISTOGRAMS ===
        print("\nProcessing data in chunks for histograms...")
        histograms = process_chunks_for_histograms(
            chunk_files, chunk_dir, numerical_columns_info, encoding_maps,ola_2, initial_ri
        )
        print("Histograms collected.")

        # === MERGE HISTOGRAMS AND CALCULATE FINAL BIN WIDTHS ===
        print("\nMerging histograms and finding final bin widths...")
        global_histogram = ola_2.merge_histograms(histograms)
        final_rf = ola_2.get_final_binwidths(global_histogram, k, l)
        supp_percent = ola_2.get_suppressed_percent(final_rf, global_histogram, k)
        lowest_dm_star = ola_2.lowest_dm_star
        num_eq_classes = ola_2.best_num_eq_classes
        eq_class_stats = ola_2.get_equivalence_class_stats(global_histogram, final_rf, k)

        logger.info(f"Final bin widths (RF): {final_rf}")
        logger.info(f"Lowest DM*: {lowest_dm_star}, EQ Classes: {num_eq_classes}, Supp%: {supp_percent:.2f}")
        log_performance(logger,"OLA_2 tree", start_time)

        print("Number of equivalence classes:", num_eq_classes)
        print("Suppressed percentage of records:", supp_percent)

        # === OPTIONALLY GENERALIZE FIRST CHUNK ===
        # Uncomment if you want to save generalized first chunk
        generalize_first_chunk(chunk_files[0], output_path,numerical_columns_info, encoding_maps, ola_2, final_rf)

        # === PIPELINE TIMING AND LOGGING ===
        elapsed_time = time.time() - start_time
        h, m, s = format_time(elapsed_time)
        print(f"\nTotal time taken: {h}h {m}m {s}s")
        log_performance(logger,"SKALD full pipeline", start_time)
        with open(log_file, "a") as f:
            f.write("=== SKALD Completed ===\n\n")

        return final_rf, elapsed_time, lowest_dm_star, num_eq_classes, eq_class_stats
    else:
        return json_output, None, None, None, None

# === ENTRY POINT ===
if __name__ == "__main__":
    #CONFIG_PATH = "config_CD.json"
    CONFIG_PATH = os.path.join('config', os.listdir('config')[0])
    print(f"Running SKALD pipeline with config: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    operations = config["operations"]
    dataset = config["data_type"]
    conf = config[dataset]
    #print("Operations to perform:", operations)
    if "chunking" in operations:
        suppress = conf.get("suppress", [])
        pseudonymize = conf.get("pseudonymize", [])
        encrypt = conf.get("encrypt", [])
        hardcoded_min_max = conf.get("hardcoded_min_max", {})
        quasi_identifiers = conf.get("quasi_identifiers", {})
        categorical = conf["quasi_identifiers"].get("categorical", [])
        numerical = conf["quasi_identifiers"].get("numerical", [])
        sizes = conf.get("bin_width_multiplication_factor", {})
        yaml_config = {
            "enable_k_anonymity": conf.get("enable_k_anonymity", True),
            "output_path": "generalized_chunk1.csv",
            "output_directory": "pipelineOutput",
            "key_directory": "keys",
            "log_file": "log.txt",
            "suppress": suppress,
            "pseudonymize": pseudonymize,
            "encrypt": encrypt,
            "quasi_identifiers": {
                "numerical": numerical,
                "categorical": categorical
            },
            "k": conf["k_anonymize"]["k"],
            "l": conf["l_diversity"]["l"],
            "sensitive_parameter": conf.get("sensitive_parameter"),
            "bin_width_multiplication_factor": sizes,
            "hardcoded_min_max": hardcoded_min_max,
            "suppression_limit": conf.get("suppression_limit", 0)
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_yaml:
            yaml.dump(
                yaml_config,
                tmp_yaml,
                sort_keys=False,
                Dumper=InlineListDumper
            )
            config_path = tmp_yaml.name
    result = run_pipeline(config_path=config_path)
    print("\nPipeline completed successfully.")
    #print("Result summary:", result)