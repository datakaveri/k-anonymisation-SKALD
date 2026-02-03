use std::collections::HashSet;
use std::fs;

use ndarray::{ArrayD, IxDyn};
use serde::{Deserialize, Serialize};
use std::time::Instant;


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
    elapsed_time: f64, 
}

/// ===============================
/// Build lattice LEVEL-BY-LEVEL
/// ===============================
fn build_lattice_levels(
    initial: &[usize],
    max_levels_pow: &[usize],
) -> Vec<Vec<Vec<usize>>> {
    let mut tree = vec![vec![initial.to_vec()]];
    let mut seen = HashSet::new();
    seen.insert(initial.to_vec());

    // max bin width = 2^level
    let max_levels: Vec<usize> =
        max_levels_pow.iter().map(|&l| 1usize << l).collect();

    loop {
        let mut next = Vec::new();

        for node in tree.last().unwrap() {
            for i in 0..node.len() {
                let mut child = node.clone();
                child[i] = (child[i] * 2).min(max_levels[i]);

                if child != *node && seen.insert(child.clone()) {
                    next.push(child);
                }
            }
        }

        if next.is_empty() {
            break;
        }
        tree.push(next);
    }

    tree
}

/// ===============================
/// Merge histogram (Python-exact)
/// ===============================
fn merge_histogram(hist: &ArrayD<i64>, rf: &[usize]) -> ArrayD<i64> {
    let ndim = hist.ndim();
    let mut out_shape = Vec::with_capacity(ndim);
    for (axis, &bw) in rf.iter().enumerate() {
        let size = hist.shape()[axis];
        let bins = (size + bw - 1) / bw;
        out_shape.push(bins);
    }

    let mut out = ArrayD::<i64>::zeros(IxDyn(&out_shape));
    let mut out_idx = vec![0usize; ndim];

    for (idx, v) in hist.indexed_iter() {
        for i in 0..ndim {
            out_idx[i] = idx[i] / rf[i];
        }
        *out.get_mut(IxDyn(&out_idx)).unwrap() += *v;
    }

    out
}

/// ===============================
/// k-anonymity + suppression
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
        (suppression_limit * total_records as f64).floor() as usize;

    suppressed <= allowed
}

/// ===============================
/// DM* metric (Python exact)
/// ===============================
fn compute_dm_star(hist: &ArrayD<i64>, k: usize) -> (i64, usize) {
    let mut dm = 0i64;
    let mut eq = 0usize;
    let mut suppressed = 0i64;

    for &v in hist.iter() {
        if (v as usize) >= k {
            dm += v * v;
            eq += 1;
        } else {
            suppressed += v;
        }
    }

    dm += suppressed * suppressed;
    (dm, eq)
}

fn main() {
    // ---------------------------
    // Load Python input
    // ---------------------------
    let input: PythonInput = serde_json::from_str(
        &fs::read_to_string("debug/python_result.json").unwrap(),
    )
    .unwrap();

    // ---------------------------
    // Load histogram
    // ---------------------------
    let shape: Vec<usize> =
        serde_json::from_str(&fs::read_to_string("debug/global_hist_shape.json").unwrap()).unwrap();
    let flat: Vec<i64> =
        serde_json::from_str(&fs::read_to_string("debug/global_hist_flat.json").unwrap()).unwrap();

    let base_hist = ArrayD::from_shape_vec(IxDyn(&shape), flat).unwrap();

    // ---------------------------
    // Build lattice
    // ---------------------------
    let start = Instant::now();
    println!("Started time ,{:?}", start);
    let tree = build_lattice_levels(&input.initial_ri, &input.max_levels);
    println!("time to build tree: {:.6} seconds", start.elapsed().as_secs_f64());
    let mut best_rf = None;
    let mut best_dm = i64::MAX;
    let mut best_eq = None;
    println!("Time to init loop: {:.6} seconds", start.elapsed().as_secs_f64());
    for rf in tree.iter().flatten() {
        let merged = merge_histogram(&base_hist, rf);
        let pass = passes_k_or_suppression(
            &merged,
            input.k,
            input.suppression_limit,
            input.total_records,
        );

        if pass {
            let (dm, eq) = compute_dm_star(&merged, input.k);
            if dm < best_dm {
                best_dm = dm;
                best_rf = Some(rf.clone());
                best_eq = Some(eq);
            }
        }
    }
    println!("Time to mark status at get best RF: {:.6} seconds", start.elapsed().as_secs_f64());
    let elapsed = start.elapsed();
    println!("Elapsed time: {:.6} seconds", elapsed.as_secs_f64());
    // ---------------------------
    // Write result
    // ---------------------------
    let result = RustResult {
        best_rf,
        lowest_dm_star: if best_dm == i64::MAX {
            None
        } else {
            Some(best_dm)
        },
        num_equivalence_classes: best_eq,
        elapsed_time: elapsed.as_secs_f64(),
    };

    fs::write(
        "debug/rust_result.json",
        serde_json::to_string_pretty(&result).unwrap(),
    )
    .unwrap();

    println!("=== RUST OLA-2 PARITY RESULT ===");
    println!("{:#?}", result);
}
