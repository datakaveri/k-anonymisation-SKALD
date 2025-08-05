
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
            "k": dataset_config.get("k", 2),
            "l": dataset_config.get("l", 1)
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
        levels[column] = 3  # make dynamic later

    # Numerical QIDs → generalize + levels + width = 1
    for num in dataset_config.get("quasi_identifiers", {}).get("numerical", []):
        column = num["column"]
        encode = num.get("encode", False)
        size = num.get("size", 2)

        min_max = config.get("hardcoded_min_max", {}).get(column)
        if not min_max:
            raise ValueError(f"Missing hardcoded_min_max for column: {column}")
        min_val, max_val = min_max

        gen_col = f"{column}_encoded" if encode else column
        generalize.append(gen_col)
        #print(f"Processing column: {gen_col} (encode={encode}, size={size})")
        width[gen_col] = 1
        levels[gen_col] = int(math.log2(max_val - min_val) + 1) # formula for calculating number of levels


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
        floored = math.floor(avg)
        print(f"   • {qi}: {floored}")
        average_row[qi] = floored

    node = [average_row[qi] for qi in qi_order]
    print("Node set to average:", node)

    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["chunk"] + qi_order)
        writer.writerow(average_row)
        print(f"Appended averages row to {csv_path}")

    return node  # Return the final averaged floor node


def calculate_DM(arx_ready_config, num_chunks, dataset_name, node):
    full_dataset_path = dataset_name
    data_folder = os.path.basename(os.path.dirname(full_dataset_path))
    k = arx_ready_config['k']

    generalize_cols = arx_ready_config.get("generalize", [])
    width_config = arx_ready_config.get("width", {})

    qis = []

    for col in generalize_cols:
        if col in width_config:
            # Numerical QI
            qis.append({
                "column_name": col,
                "is_categorical": False,
                "min_value": 0  # or handle dynamically
            })
        else:
            # Categorical QI
            qis.append({
                "column_name": col,
                "is_categorical": True
            })

    #n = int(input("number of chunks: "))
    n = num_chunks

    # Step 1: First pass to find min/max for numeric columns
    global_minmax = {}
    for q in qis:
        if not q['is_categorical']:
            global_minmax[q['column_name']] = [np.inf, -np.inf]

    # First lightweight scan
    for i in range(1, 101):
        if n <= 0:
            break
        file_path = os.path.join("small_data", f"small_chunk{i}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, usecols=[q['column_name'] for q in qis], dtype={
                'Age': np.float32,
                'PIN Code_encoded': np.int32
            })
            for col in global_minmax:
                global_minmax[col][0] = min(global_minmax[col][0], df[col].min())
                global_minmax[col][1] = max(global_minmax[col][1], df[col].max())
            n -= 1

    # Step 2: Now second pass for DM* computation
    histogram = Counter()
    #n = int(input("\nnumber of chunks again (for second pass): "))  # Ask again since `n` got 0 above
    n = num_chunks

    for i in range(1, 125):
        if n <= 0:
            break
        file_path = os.path.join("small_data", f"small_chunk{i}.csv")
        if os.path.exists(file_path):
            print(f"\n📂 Processing chunk: {file_path}")
            df = pd.read_csv(file_path, dtype={
                'Age': np.float32,
                'PIN Code_encoded': np.int32
            })
            df["Blood Group"] = df["Blood Group"].astype('category')
            df["Gender"] = df["Gender"].astype('category')

            generalized_cols = []
            for (col_info, bw) in zip(qis, node):
                col_name = col_info['column_name']
                is_cat = col_info['is_categorical']
                if is_cat:
                    min_val, max_val = None, None
                else:
                    min_val, max_val = global_minmax[col_name]

                generalized = generalize_column(
                    df[col_name],
                    col_name,
                    bw if is_cat else 0,
                    min_val, max_val,
                    bw if not is_cat else None
                )
                generalized_cols.append(generalized)

            df['__eq_class__'] = list(zip(*generalized_cols))
            chunk_histogram = df['__eq_class__'].value_counts().to_dict()
            histogram.update(chunk_histogram)

            n -= 1

    # Step 3: Calculate DM*
    print("\n🔍 Sorted Equivalence Classes (Top 20 shown):")
    i = 2
    for eq_class, count in sorted(histogram.items(), key=lambda item: item[1], reverse=True)[:20]:
        if count >= k:
            print(i, f"{eq_class} -> {count}")
            i += 1

    low_count_sum = sum(count for count in histogram.values() if count < k)
    dm_star = sum(count * count for count in histogram.values() if count >= k)
    dm_star += low_count_sum * low_count_sum
    arx_eq_classes = len(histogram)
    print(f"\n Total Equivalence Classes: {arx_eq_classes}")

    print("\n⚠️ Low count total:", low_count_sum)
    print(f"\n⭐ Global DM* value: {dm_star}")
    return dm_star,arx_eq_classes