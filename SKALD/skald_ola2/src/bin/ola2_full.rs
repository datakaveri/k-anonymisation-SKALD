use std::fs;

use ndarray::{ArrayD, Axis, IxDyn};
use serde::{Deserialize, Serialize};

/// ===============================
/// Python input (from debug file)
/// ===============================
#[derive(Debug, Deserialize)]
struct PythonInput {
    initial_ri: Vec<usize>,
    max_levels: Vec<usize>,
    k: usize,
    suppression_limit: f64, // percent
    total_records: usize,
}

/// ===============================
/// Rust output (to debug file)
/// ===============================
#[derive(Debug, Serialize)]
struct RustResult {
    best_rf: Option<Vec<usize>>,
    lowest_dm_star: Option<i64>,
    num_equivalence_classes: Option<usize>,
}

/// ===============================
/// Build lattice LEVEL-BY-LEVEL
/// (matches Python build_tree)
/// ===============================
fn build_lattice_levels(
    initial: &[usize],
    max_levels_pow: &[usize],
) -> Vec<Vec<Vec<usize>>> {
    let mut tree: Vec<Vec<Vec<usize>>> = vec![vec![initial.to_vec()]];
    let mut seen = std::collections::HashSet::new();
    seen.insert(initial.to_vec());
    let max_levels: Vec<usize> = max_levels_pow
    .iter()
    .map(|&lvl| 1usize << lvl)
    .collect();



    loop {
        let mut next_level = Vec::new();

        for node in tree.last().unwrap() {
            for i in 0..node.len() {
                let mut new_node = node.clone();
                new_node[i] = (new_node[i] * 2).min(max_levels[i]);

                if new_node != *node && !seen.contains(&new_node) {
                    seen.insert(new_node.clone());
                    next_level.push(new_node);
                }
            }
        }

        if next_level.is_empty() {
            break;
        }

        tree.push(next_level);
    }

    tree
}

/// ===============================
/// Merge histogram (Python-equivalent)
/// ===============================
fn merge_histogram(hist: &ArrayD<i64>, rf: &[usize]) -> ArrayD<i64> {
    let mut merged = hist.clone();

    for (axis, &group) in rf.iter().enumerate() {
        let size = merged.shape()[axis];
        let pad = (group - (size % group)) % group;

        if pad > 0 {
            let mut new_shape = merged.shape().to_vec();
            new_shape[axis] += pad;

            let mut padded = ArrayD::<i64>::zeros(IxDyn(&new_shape));
            for (idx, v) in merged.indexed_iter() {
                padded[idx.clone()] = *v;
            }
            merged = padded;
        }

        let mut reshaped = merged.shape().to_vec();
        reshaped[axis] /= group;
        reshaped.insert(axis + 1, group);

        merged = merged
            .into_shape(IxDyn(&reshaped))
            .unwrap()
            .sum_axis(Axis(axis + 1));
    }

    merged
}


/// ===============================
/// Python-exact k + suppression logic
/// ===============================
fn passes_k_or_suppression(
    hist: &ArrayD<i64>,
    k: usize,
    suppression_limit: f64,
    total_records: usize,
) -> bool {
    let mut suppressed = 0usize;

    for &v in hist.iter() {
        if v > 0 && (v as usize) < k {
            suppressed += v as usize;
        }
    }

    let allowed =
        (suppression_limit * total_records as f64 / 100.0).floor() as usize;

    suppressed <= allowed
}

/// ===============================
/// Compute DM* + eq classes
/// ===============================
fn compute_dm_star(hist: &ArrayD<i64>, k: usize) -> (i64, usize) {
    let mut dm = 0i64;
    let mut eq = 0usize;
    let mut sup = 0i64;

    for &v in hist.iter() {
        if (v as usize) >= k {
            dm += v * v;
            eq += 1;
        }
        else {
            sup += v;
        }
    }
    dm += sup * sup;
    eq += 1;
    (dm, eq)
}

fn main() {
    // ---------------------------
    // Load Python input
    // ---------------------------
    let input: PythonInput = serde_json::from_str(
        &fs::read_to_string("debug/python_result.json")
            .expect("Missing python_result.json"),
    )
    .expect("Invalid python_result.json");

    // ---------------------------
    // Load histogram
    // ---------------------------
    let shape: Vec<usize> = serde_json::from_str(
        &fs::read_to_string("debug/global_hist_shape.json").unwrap(),
    )
    .unwrap();

    let flat: Vec<i64> = serde_json::from_str(
        &fs::read_to_string("debug/global_hist_flat.json").unwrap(),
    )
    .unwrap();

    let base_hist =
        ArrayD::from_shape_vec(IxDyn(&shape), flat).unwrap();

    // ---------------------------
    // Build lattice (LEVEL ORDER)
    // ---------------------------
    let tree = build_lattice_levels(
        &input.initial_ri,
        &input.max_levels,
    );

    // ---------------------------
    // Evaluate all nodes (Python logic)
    // ---------------------------
    let mut pass_nodes = Vec::new();
    println!("max levels: {:?}", input.max_levels);
    println!("Lattice : {:?}", tree);
    for level in &tree {
        for rf in level {
            let merged = merge_histogram(&base_hist, rf);

            if passes_k_or_suppression(
                &merged,
                input.k,
                input.suppression_limit,
                input.total_records,
            ) {
                pass_nodes.push(rf.clone());
            }
        }
    }

    // ---------------------------
    // Select best RF via DM*
    // ---------------------------
    let mut best_rf = None;
    let mut best_dm = i64::MAX;
    let mut best_eq = None;

    for rf in pass_nodes {
        let merged = merge_histogram(&base_hist, &rf);
        let (dm, eq) = compute_dm_star(&merged, input.k);

        if dm < best_dm {
            best_dm = dm;
            best_rf = Some(rf);
            best_eq = Some(eq);
        }
    }

    // ---------------------------
    // Write Rust result
    // ---------------------------
    let result = RustResult {
        best_rf,
        lowest_dm_star: if best_dm == i64::MAX {
            None
        } else {
            Some(best_dm)
        },
        num_equivalence_classes: best_eq,
    };

    fs::write(
        "debug/rust_result.json",
        serde_json::to_string_pretty(&result).unwrap(),
    )
    .unwrap();

    println!("=== RUST OLA-2 PARITY RESULT ===");
    println!("{:#?}", result);
}
