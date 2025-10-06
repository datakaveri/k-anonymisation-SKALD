
import requests, csv, json, time, math
import os
import pandas as pd
import numpy as np
from collections import Counter

# Categorical Hierarchies

blood_group_hierarchy = {
    1: lambda x: x,
    2: lambda x: {
    "A+": "A", "A-": "A", "B+": "B", "B-": "B",
    "AB+": "AB", "AB-": "AB", "O+": "O", "O-": "O"
    }.get(x, x),
    3: lambda x: "*"
}

profession_hierarchy = {
    0: lambda x: x,
    1: lambda x: {
        "Medical Specialists": "Healthcare", "Allied Health": "Healthcare", "Nursing": "Healthcare", "Healthcare Support": "Healthcare",
        "K-12 Education Teacher": "Education", "Higher Education Teacher": "Education", "Supplemental Education Teacher": "Education", "University Professor": "Education",
        "Performing Arts": "Creative", "Visual & Media Arts": "Creative", "Design": "Creative", "Mixed Media Artist": "Creative",
        "Traditional Engineering": "Engineering", "Software Engineering": "Engineering", "Data & Analytics": "Engineering", "AI & Machine Learning": "Engineering"
    }.get(x, x),
    2: lambda x: {
        "Medical Specialists": "Service Sector", "Allied Health": "Service Sector", "Nursing": "Service Sector", "Healthcare Support": "Service Sector",
        "K-12 Education Teacher": "Service Sector", "Higher Education Teacher": "Service Sector", "Supplemental Education Teacher": "Service Sector", "University Professor": "Service Sector",
        "Performing Arts": "Non-Service", "Visual & Media Arts": "Non-Service", "Design": "Non-Service", "Mixed Media Artist": "Non-Service",
        "Traditional Engineering": "Non-Service", "Software Engineering": "Non-Service", "Data & Analytics": "Non-Service", "AI & Machine Learning": "Non-Service"
    }.get(x, x),
    3: lambda x: "*"
}

gender_hierarchy = {
                1: {  # Level 0: No generalization
                    "Male": "Male", "Female" :"Female"
                },
                2: {  # Level 1: Fully generalized
                    "Male": "*", "Female" :"*"
                }
}

def convert_numpy(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
def convert_ui_config_to_arx(config):
    data_type = config["data_type"]
    dataset_config = config.get(data_type, {})
    #print(dataset_config)
    #print(list(set(dataset_config.get("suppress", []) + dataset_config.get("pseudonymize", []))))
    arx_config = {
        "num_chunks": dataset_config.get("number_of_chunks"),
        "dataset_name": dataset_config.get("dataset_name", "default_dataset"),
        "suppress": dataset_config.get("suppress", []),
        "pseudonymize": dataset_config.get("pseudonymize", []),
        "sensitive_column": dataset_config.get("sensitive_column", []),
        "insensitive_columns": dataset_config.get("insensitive_columns", []),
        "generalize": [],  # Fill based on categorical/num QIDs
        "levels": {},
        "width": {},
        "k_anonymize": {
            "k": dataset_config.get("k_anonymize", {}).get("k"),
            "l": dataset_config.get("l_diversity", {}).get("l")
        },
        "suppression_limit": dataset_config.get("suppression_limit", 0.01)
    }    

    generalize = []
    width = {}
    levels = {}

    # Categorical QIDs → generalize + levels (NO width)
    for cat in dataset_config.get("quasi_identifiers", {}).get("categorical", []):
        column = cat["column"]
        generalize.append(column)
        if column == "Blood Group":
            levels[column] = 3
        else:
            levels[column] = 2

    # Numerical QIDs → generalize + levels + width = 1
    for num in dataset_config.get("quasi_identifiers", {}).get("numerical", []):
        column = num["column"]
        encode = num.get("encode", False)
        size = num.get("size", 2)

        min_max = config.get("hardcoded_min_max", {}).get(column)
        if not min_max:
            raise ValueError(f"Missing hardcoded_min_max for column: {column}")
        min_val, max_val = min_max
        print(f"Processing column: {column} (encode={encode}, size={size}, min={min_val}, max={max_val})")
        #gen_col = f"{column}_encoded" if encode else column
        gen_col = column
        generalize.append(gen_col)
        #print(f"Processing column: {gen_col} (encode={encode}, size={size})")
        width[gen_col] = 1
        levels[gen_col] = int(math.log2(max_val - min_val) + 1) # formula for calculating number of levels

    generalize.sort()  # Sort to ensure consistent order
    arx_config["generalize"] = generalize
    arx_config["levels"] = levels
    arx_config["width"] = width  # width will only include numerical QIDs

    return arx_config

# Generalization function
def generalize_column(series, column_name, level, min_val=None, max_val=None, bin_width=None):
    if column_name == "Blood Group":
        return series.map(blood_group_hierarchy[level])
    elif column_name == "Gender":
        return series.map(gender_hierarchy[level])
    else:
        #print(min_val, max_val, bin_width)
        bin_edges = np.arange(min_val, max_val + bin_width + 1, bin_width)
        labels = [f"[{int(bin_edges[i])}-{int(bin_edges[i+1]-1)}]" for i in range(len(bin_edges)-1)]
        return pd.cut(series, bins=bin_edges, labels=labels, include_lowest=True)

def average_best_node(arx_ready_config, num_chunks, url, headers, base_chunk_path, params_template):
    qi_order        = arx_ready_config['generalize']
    num_chunks      = num_chunks
    widths          = arx_ready_config['width']      # numeric QIs
    gen_level_sums  = {qi: 0 for qi in qi_order}
    categorical_qis = [qi for qi in qi_order if qi not in widths]
    
    csv_path = "best_nodes.csv"

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["chunk"] + qi_order)
        writer.writeheader()

        for i in range(1, num_chunks + 1):
            chunk_path = base_chunk_path.format(i=i)
            params = params_template.copy()
            params["dataset_name"] = os.path.abspath(chunk_path)
            #print(f"\nChunk {i} request: {params['dataset_name']}")
            #print(type(params))
            r = requests.post(url, headers=headers, data=json.dumps(params, default=convert_numpy))

            try:
                data = r.json()
            except Exception:
                print("Response is not valid JSON:", r.text)
                continue

            # Extract rows
            if isinstance(data, list):
                rows = data
            elif isinstance(data, dict):
                lists = [v for v in data.values() if isinstance(v, list)]
                if not lists:
                    print("No list found in response; can't extract rows.")
                    continue
                rows = lists[0]
            else:
                print("Unexpected response structure")
                continue

            transformation_node = None
            for row in rows:
                if isinstance(row, dict) and row.get("meta") is True:
                    transformation_node = row.get("transformation_node")
                    break

            if transformation_node is None:
                print(f"No transformation node found in chunk {i}")
                continue

            if isinstance(transformation_node, str):
                transformation_node = json.loads(transformation_node)
            if not isinstance(transformation_node, list):
                print("transformation node is not a list:", transformation_node)
                continue

            print("Best node for chunk", i, "→", transformation_node)

            row = {"chunk": i}
            for idx, qi in enumerate(qi_order):
                lvl = transformation_node[idx]
                if qi in categorical_qis:
                    lvl += 1
                row[qi] = lvl
                gen_level_sums[qi] += lvl

            writer.writerow(row)
            csvfile.flush()
            time.sleep(1)

    print(f"\nAverage generalization levels across {num_chunks} chunks (floored):")
    average_row = {"chunk": "average"}
    for qi in qi_order:
        avg = gen_level_sums[qi] / num_chunks
        #floored = math.floor(avg)
        floored = round(avg)
        print(f"   • {qi}: {floored}")
        average_row[qi] = floored

    node = [average_row[qi] for qi in qi_order]
    print("Node set to average:", node)

    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["chunk"] + qi_order)
        writer.writerow(average_row)
        print(f"Appended averages row to {csv_path}")

    return node  # Return the final averaged floor node

def generalize_pincode(series, bw):
    """Generalize PIN codes based on digit masking (bw = number of stars)."""
    generalized_labels = []
    for pin in series:
        pin = int(pin)
        # divisor depends on bw → 10^bw
        divisor = 10 ** bw
        start = (pin // divisor) * divisor + 1
        end = start + divisor - 1
        generalized_labels.append(f"[{start}-{end}]")
    return pd.Series(generalized_labels, index=series.index)

def calculate_DM(config, arx_ready_config, num_chunks, dataset_name, node):
    full_dataset_path = dataset_name
    data_folder = os.path.basename(os.path.dirname(full_dataset_path))
    k = arx_ready_config['k']
    l = arx_ready_config['k_anonymize']['l']
    print(f"\nCalculating DM* for dataset: {full_dataset_path} with k={k}, l={l}, node={node}")
    sensitive_col = "Health Condition"
    generalize_cols = arx_ready_config.get("generalize", [])
    width_config = arx_ready_config.get("width", {})
    qis = []

    # Build QI info
    for col in generalize_cols:
        if col in width_config:
            qis.append({
                "column_name": col,
                "is_categorical": False
            })
        else:
            qis.append({
                "column_name": col,
                "is_categorical": True
            })

    n = num_chunks
    histogram = Counter()
    sensitive_sets ={}
    for i in range(1, 125):
        if n <= 0:
            break
        file_path = os.path.join("5K_rows_data", f"5krows_chunk{i}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, dtype={
                'Age': np.int32,
                'PIN Code': np.int32,
                'BMI': np.float32,
            })
            df["Blood Group"] = df["Blood Group"].astype('category')
            df["Gender"] = df["Gender"].astype('category')


            generalized_cols = []
 
            for (col_info, bw) in zip(qis, node):
                col_name = col_info['column_name']
                is_cat = col_info['is_categorical']

                if is_cat:
                    generalized = generalize_column(df[col_name],col_name, bw)
                elif col_name == "PIN Code":
                    generalized = generalize_pincode(df[col_name], bw)
                else:
                    if bw == 0:
                        bw = 1
                    col_data = df[col_name]
                    min_val, max_val = col_data.min(), col_data.max()

                    if pd.api.types.is_integer_dtype(col_data):
                        # integer bins
                        bin_edges = list(range(int(min_val), int(max_val) + bw, bw))
                        labels = [
                            f"[{bin_edges[j]}-{bin_edges[j+1]-1}]"
                            for j in range(len(bin_edges)-1)
                        ]
                        #print(labels)
                    else:
                        # float bins
                        bin_edges = np.arange(min_val, max_val + bw, bw)
                        labels = [
                            f"[{round(bin_edges[j], 2)}-{round(bin_edges[j+1], 2)}]"
                            for j in range(len(bin_edges)-1)
                        ]
                        #print(labels)
                    generalized = pd.cut(
                        col_data, bins=bin_edges, labels=labels,
                        include_lowest=True, right=False
                    )

                generalized_cols.append(generalized)

            df['__eq_class__'] = list(zip(*generalized_cols))
            for eq_class, group in df.groupby('__eq_class__'):
                count = len(group)
                histogram[eq_class] += count
                if sensitive_col:
                    if eq_class not in sensitive_sets:
                        sensitive_sets[eq_class] = set()
                    sensitive_sets[eq_class].update(group[sensitive_col].unique())

            n -= 1
        else:
            print(f"Chunk file not found: {file_path}")
            continue

    # Step 3: Calculate DM*
    #print("\n🔍 Sorted Equivalence Classes (Top 20 shown):")
    i = 1
    for eq_class, count in sorted(histogram.items(), key=lambda item: item[1], reverse=True)[:48]:
        diversity = len(sensitive_sets.get(eq_class, []))
        print(i, f"{eq_class} -> {count}, diversity={diversity}")
        i += 1

    low_count_sum = 0
    dm_star = 0
    arx_eq_classes = 0

    for eq_class, count in histogram.items():
        diversity = len(sensitive_sets.get(eq_class, [])) if sensitive_col else 1
        #print(f"Equivalence Class: {eq_class} | Count: {count} | Diversity: {diversity}")
        if count < k or diversity < l:  # suppress
            low_count_sum += count
        else:
            dm_star += count * count
            arx_eq_classes += 1
    dm_star += low_count_sum * low_count_sum
    arx_eq_classes += 1

    print(f"\n Total Equivalence Classes: {arx_eq_classes}")
    print("\n⚠️ Low count (suppressed) total:", low_count_sum)
    print(f"\n⭐ Global DM* value: {dm_star}")

    return dm_star, arx_eq_classes