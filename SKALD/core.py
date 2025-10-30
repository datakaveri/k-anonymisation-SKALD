import os
import time
import logging
import psutil
import pandas as pd

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
    
    # Override config parameters if explicitly provided
    if k is not None:
        config.k = k
    if chunks is not None:
        config.number_of_chunks = chunks
    if chunk_dir is not None:
        config.chunk_directory = chunk_dir

    # === PIPELINE PARAMETERS ===
    n = config.number_of_chunks
    k = config.k
    l = config.l
    chunk_dir = config.chunk_directory
    output_path = config.output_path
    suppression_limit = config.suppression_limit

    # Estimate memory-based limit for OLA tree
    process = psutil.Process(os.getpid())
    available_ram_bytes = psutil.virtual_memory().available 
    max_equivalence_classes = available_ram_bytes // 10000
    logger.info(f"Available RAM: {available_ram_bytes}, maximum number of equivalence classes: {max_equivalence_classes}")

    # Columns and settings from config
    suppressed_columns = config.suppress
    pseudonymized_columns = config.pseudonymize
    sensitive_parameter = config.sensitive_parameter
    save_output = config.save_output
    categorical_columns = [cat_qi.column for cat_qi in config.quasi_identifiers.categorical]

    numerical_columns_info = [
        {"column": num_qi.column, "encode": num_qi.encode, "type": num_qi.type}
        for num_qi in config.quasi_identifiers.numerical
    ]

    hardcoded_min_max = config.hardcoded_min_max
    multiplication_factors = config.bin_width_multiplication_factor

    # === VALIDATE CHUNK DIRECTORY ===
    if not os.path.exists(chunk_dir):
        raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")

    all_files = sorted([f for f in os.listdir(chunk_dir) if f.endswith(".csv")])
    if not all_files:
        raise ValueError("No CSV files found in the chunk directory.")
    chunk_files = all_files[:n]
    print(f"Processing {n} chunks from {chunk_dir}...")

    start_time = time.time()

    # === ENCODE NUMERICAL COLUMNS ACROSS ALL CHUNKS ===
    encoding_maps = encode_numerical_columns(
        chunk_files, chunk_dir, numerical_columns_info,
        suppressed_columns, pseudonymized_columns
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
        chunk_files, chunk_dir, numerical_columns_info, encoding_maps,
        suppressed_columns, pseudonymized_columns, ola_2, initial_ri
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
    #generalize_first_chunk(chunk_files[0], chunk_dir, numerical_columns_info, encoding_maps,
    #                       suppressed_columns, pseudonymized_columns, ola_2, final_rf, output_path)

    # === PIPELINE TIMING AND LOGGING ===
    elapsed_time = time.time() - start_time
    h, m, s = format_time(elapsed_time)
    print(f"\nTotal time taken: {h}h {m}m {s}s")
    log_performance(logger,"SKALD full pipeline", start_time)
    with open(log_file, "a") as f:
        f.write("=== SKALD Completed ===\n\n")

    return final_rf, elapsed_time, lowest_dm_star, num_eq_classes, eq_class_stats
