//! OLA (Optimal Lattice Anonymization) algorithm — phases 1 and 2.
//!
//! # OLA-1
//! Performs a fast **synthetic** search over a generalization tree built from
//! QI descriptors alone (no data read). Finds the minimal node that keeps the
//! number of equivalence classes below a configurable RAM limit (`max_eq`).
//! The result becomes `initial_ri` — the base generalization used to bin the
//! real data into the sparse histogram for OLA-2.
//!
//! # OLA-2
//! Given the real [`SparseHist`] histogram, runs a **binary lattice search**:
//! each iteration evaluates the middle unmarked node at the median unmarked
//! level, then uses monotonicity to mark entire subtrees (pass → all
//! descendants pass) and ancestor chains (fail → all ancestors fail).
//! Returns the node with minimum DM* that satisfies k-anonymity plus the
//! optional suppression fraction limit.
//!
//! Helper functions for merging the histogram, scoring nodes, and propagating
//! pass/fail status through the tree are also defined here.

use crate::pipeline::bootstrap::{split_csv_line_basic, validation, PipelineError};
use super::{QuasiIdentifierLite, SparseHist, Ola2NodeScore, Ola2SearchResult, base_col_name};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

// ── Arithmetic helpers ────────────────────────────────────────────────────────

/// Integer ceiling division for non-negative numerator and positive denominator.
///
/// Negative or zero inputs are clamped: `a` → `max(a,0)`, `b` → `max(b,1)`.
///
/// # Arguments
/// * `a` — numerator.
/// * `b` — denominator.
///
/// # Returns
/// `⌈a / b⌉` as `i64`.
fn ceil_div_i64(a: i64, b: i64) -> i64 {
    let aa = a.max(0);
    let bb = b.max(1);
    (aa + bb - 1) / bb
}

// ── Categorical level bounds ──────────────────────────────────────────────────

/// Returns the maximum generalization level for a categorical QI column in OLA-1.
///
/// OLA-1 uses a fixed hierarchy depth per supported category so that the
/// synthetic equivalence-class count stays predictable.
///
/// Supported columns and their depths:
/// - `"blood group"` → 3 (exact / ABO / `*`)
/// - `"gender"` → 2 (exact / `*`)
/// - `"profession"` → 4 (exact / sector / super-sector / `*`)
///
/// # Errors
/// Returns [`PipelineError`] with code `GENERALIZATION_FAILED` for any
/// unsupported categorical column name.
fn max_categorical_level_ola1(col: &str) -> Result<i64, PipelineError> {
    match col.trim().to_lowercase().as_str() {
        "blood group" => Ok(3),
        "gender" => Ok(2),
        "profession" => Ok(4),
        _ => Err(validation(
            "GENERALIZATION_FAILED",
            "Unsupported categorical column in OLA-1",
            col,
        )),
    }
}

/// Returns the maximum generalization level for a categorical QI column in OLA-2.
///
/// Unlike [`max_categorical_level_ola1`], OLA-2 defaults to level 4 for
/// unknown columns so that the lattice is always finite.
///
/// # Arguments
/// * `col` — column name (case-insensitive).
fn max_categorical_level_ola2(col: &str) -> i64 {
    match col.trim().to_lowercase().as_str() {
        "blood group" => 3,
        "gender" => 2,
        _ => 4,
    }
}

// ── OLA-1 equivalence-class estimator ────────────────────────────────────────

/// Estimates the number of equivalence classes produced by a given OLA-1 node.
///
/// The estimate is a product over all QIs:
/// - **Categorical**: `⌊max_level / bin_width⌋` distinct categories.
/// - **Numerical**: `⌈range / bin_width⌉` buckets.
///
/// The product is computed with saturation to avoid integer overflow.
///
/// # Arguments
/// * `qis` — quasi-identifier descriptors (length must equal `bin_widths`).
/// * `bin_widths` — generalization factors for each QI at the candidate node.
///
/// # Returns
/// Estimated equivalence class count (≥ 1).
///
/// # Errors
/// Returns [`PipelineError`] if lengths differ or a categorical column is
/// unsupported by OLA-1.
fn calculate_equivalence_classes_ola1(
    qis: &[QuasiIdentifierLite],
    bin_widths: &[i64],
) -> Result<i64, PipelineError> {
    if bin_widths.len() != qis.len() {
        return Err(validation("GENERALIZATION_FAILED", "bin width length mismatch", "OLA-1"));
    }
    let mut classes = 1_i64;
    for (qi, bw) in qis.iter().zip(bin_widths.iter()) {
        let bw = (*bw).max(1);
        let bins = if qi.is_categorical {
            let max_level = max_categorical_level_ola1(&qi.column_name)?;
            (max_level / bw).max(1)
        } else {
            ceil_div_i64((qi.get_range().ceil() as i64).max(1), bw)
        };
        classes = classes.saturating_mul(bins.max(1));
    }
    Ok(classes.max(1))
}

// ── Lattice tree builder ──────────────────────────────────────────────────────

/// Builds all levels of the generalization lattice as a list of node vectors.
///
/// Starting from `initial`, each level is obtained by "climbing" each QI by
/// one step (categorical: increment level, numerical: multiply bin width by
/// `size_factors[base_col]`). Duplicate nodes are deduplicated via a `BTreeSet`.
/// The loop terminates when no new nodes are reachable.
///
/// # Arguments
/// * `qis` — quasi-identifier descriptors.
/// * `initial` — starting node (one factor per QI, length must equal `qis`).
/// * `size_factors` — multiplication factors keyed by base column name.
/// * `ola1_mode` — `true` to use OLA-1 categorical level bounds; `false` for OLA-2.
///
/// # Returns
/// `tree[level_index]` is the list of all distinct nodes reachable at that level.
///
/// # Errors
/// Returns [`PipelineError`] on length mismatch or missing size factor.
fn build_tree_levels(
    qis: &[QuasiIdentifierLite],
    initial: &[i64],
    size_factors: &BTreeMap<String, i64>,
    ola1_mode: bool,
) -> Result<Vec<Vec<Vec<i64>>>, PipelineError> {
    if qis.len() != initial.len() {
        return Err(validation("GENERALIZATION_FAILED", "initial node length mismatch", "tree"));
    }
    let mut tree = vec![vec![initial.to_vec()]];
    let mut seen: BTreeSet<Vec<i64>> = BTreeSet::new();
    seen.insert(initial.to_vec());

    loop {
        let mut next_level: Vec<Vec<i64>> = Vec::new();
        let prev = tree.last().cloned().unwrap_or_default();
        for node in prev {
            for (i, qi) in qis.iter().enumerate() {
                let mut new_node = node.clone();
                if qi.is_categorical {
                    let max_level = if ola1_mode {
                        max_categorical_level_ola1(&qi.column_name)?
                    } else {
                        max_categorical_level_ola2(&qi.column_name)
                    };
                    if new_node[i] < max_level {
                        new_node[i] += 1;
                    }
                } else {
                    let base = base_col_name(&qi.column_name);
                    let factor = size_factors.get(&base).copied().ok_or_else(|| {
                        validation(
                            "GENERALIZATION_FAILED",
                            "Missing multiplication factor for numeric QI",
                            &base,
                        )
                    })?;
                    let max_val = (qi.get_range().ceil() as i64).max(1);
                    new_node[i] = (new_node[i] * factor.max(1)).min(max_val).max(1);
                }
                if new_node != node && seen.insert(new_node.clone()) {
                    next_level.push(new_node);
                }
            }
        }
        if next_level.is_empty() {
            break;
        }
        tree.push(next_level);
    }

    Ok(tree)
}

// ── OLA-1 public interface ────────────────────────────────────────────────────

/// Finds the minimal OLA-1 generalization node that satisfies the memory-based
/// equivalence-class constraint.
///
/// Builds the full synthetic lattice tree, evaluates each node independently
/// (no data required — only QI descriptors), and returns the lexicographically
/// smallest passing node.
///
/// # Arguments
/// * `qis` — quasi-identifier descriptors (non-empty).
/// * `_n_chunks` — unused; reserved for future adaptive sizing.
/// * `max_eq` — upper bound on the number of equivalence classes (must be > 0).
/// * `size_factors` — multiplication factors keyed by base column name.
///
/// # Returns
/// A node vector (one bin-width per QI) representing `initial_ri`.
///
/// # Errors
/// - `ANON_NO_QIS` — `qis` is empty.
/// - `GENERALIZATION_FAILED` — `max_eq ≤ 0`.
/// - `ANON_INFEASIBLE` — no node satisfies the constraint (try increasing RAM
///   budget, reducing size factors, or reducing the number of QIs).
pub fn find_ola1_initial_ri(
    qis: &[QuasiIdentifierLite],
    _n_chunks: i64,
    max_eq: i64,
    size_factors: &BTreeMap<String, i64>,
) -> Result<Vec<i64>, PipelineError> {
    if qis.is_empty() {
        return Err(validation("ANON_NO_QIS", "No quasi-identifiers defined — cannot run OLA-1", "Add at least one QI to quasi_identifiers in config"));
    }
    if max_eq <= 0 {
        return Err(validation("GENERALIZATION_FAILED", "max_eq must be > 0", "OLA-1"));
    }

    let base = vec![1_i64; qis.len()];
    let tree = build_tree_levels(qis, &base, size_factors, true)?;
    let mut node_status: BTreeMap<Vec<i64>, Option<bool>> = BTreeMap::new(); // true=pass, false=fail
    for level in &tree {
        for node in level {
            node_status.entry(node.clone()).or_insert(None);
        }
    }

    // Single pass — each node is independent (calculate_equivalence_classes_ola1 depends only
    // on the node vector itself, not on any other node's status), so no outer loop is needed.
    for level in &tree {
        for node in level {
            let classes = calculate_equivalence_classes_ola1(qis, node)?;
            node_status.insert(node.clone(), Some(classes <= max_eq));
        }
    }

    let mut passing: Vec<Vec<i64>> = node_status
        .into_iter()
        .filter_map(|(n, st)| if st == Some(true) { Some(n) } else { None })
        .collect();
    passing.sort();
    passing.into_iter().next().ok_or_else(|| {
        validation(
            "ANON_INFEASIBLE",
            "No generalization satisfies the equivalence class memory constraint (OLA-1)",
            "Increase available RAM, reduce size factors, or reduce the number of QIs",
        )
    })
}

// ── Histogram builder ─────────────────────────────────────────────────────────

/// Builds the sparse histogram from all chunk CSV files at the `initial_ri` granularity.
///
/// **Two-pass algorithm:**
/// 1. Scan all chunks to collect the complete categorical domain for each
///    categorical QI (so that domain indices are consistent even when a value
///    appears only in a subset of chunks).
/// 2. Re-scan all chunks, bin each record into a QI index-tuple, and increment
///    the histogram counter.
///
/// # Arguments
/// * `chunk_paths` — paths to the pre-processed chunk CSV files (non-empty).
/// * `qis` — quasi-identifier descriptors (length must equal `initial_ri`).
/// * `initial_ri` — OLA-1 bin widths used to bucket numerical values.
///
/// # Returns
/// `(histogram, total_valid_records)`.
///
/// # Errors
/// - `GENERALIZATION_FAILED` — length mismatch, unreadable file, missing QI
///   column, or no valid histogram records produced.
/// - `DATA_COLUMN_MISSING` — a QI column (or its base name) is absent from the
///   CSV header.
pub fn build_sparse_histogram(
    chunk_paths: &[PathBuf],
    qis: &[QuasiIdentifierLite],
    initial_ri: &[i64],
) -> Result<(SparseHist, i64), PipelineError> {
    if qis.len() != initial_ri.len() {
        return Err(validation("GENERALIZATION_FAILED", "initial_ri length mismatch", "OLA-2"));
    }

    // Resolve column indices from the first chunk's header (schema is identical across all chunks)
    let col_idx: Vec<usize> = {
        let first = chunk_paths
            .first()
            .ok_or_else(|| validation("GENERALIZATION_FAILED", "No chunks provided", "histogram"))?;
        let f = fs::File::open(first)?;
        let mut lines = BufReader::new(f).lines();
        let header = lines
            .next()
            .ok_or_else(|| validation("GENERALIZATION_FAILED", "chunk missing header", &first.display().to_string()))?
            .map_err(|e| validation("GENERALIZATION_FAILED", "failed reading chunk header", &e.to_string()))?;
        let headers = split_csv_line_basic(&header);
        let mut out = Vec::with_capacity(qis.len());
        for qi in qis {
            let fallback = base_col_name(&qi.column_name);
            let idx = headers
                .iter()
                .position(|h| h == &qi.column_name)
                .or_else(|| headers.iter().position(|h| h == &fallback))
                .ok_or_else(|| validation("DATA_COLUMN_MISSING", "Quasi-identifier column not found in CSV header", &qi.column_name))?;
            out.push(idx);
        }
        out
    };

    // Pass 1: Build categorical domains from ALL chunks so no value is missed
    let mut cat_sets: Vec<BTreeSet<String>> = vec![BTreeSet::new(); qis.len()];
    for chunk in chunk_paths {
        let f = fs::File::open(chunk)?;
        let mut lines = BufReader::new(f).lines();
        let _ = lines.next(); // skip header
        for line in lines {
            let line = line.map_err(|e| validation("GENERALIZATION_FAILED", "failed reading chunk row", &e.to_string()))?;
            if line.trim().is_empty() {
                continue;
            }
            let fields = split_csv_line_basic(&line);
            for i in 0..qis.len() {
                if qis[i].is_categorical && col_idx[i] < fields.len() {
                    cat_sets[i].insert(fields[col_idx[i]].trim().to_string());
                }
            }
        }
    }
    let categorical_domains: Vec<Vec<String>> =
        cat_sets.into_iter().map(|s| s.into_iter().collect()).collect();

    // Pass 2: Build histogram from all chunks using the complete domain sets
    let mut hist: SparseHist = BTreeMap::new();
    let mut total_records: i64 = 0;

    for chunk in chunk_paths {
        let f = fs::File::open(chunk)?;
        let mut lines = BufReader::new(f).lines();
        let _ = lines.next(); // skip header

        for line in lines {
            let line = line.map_err(|e| validation("GENERALIZATION_FAILED", "failed reading chunk row", &e.to_string()))?;
            if line.trim().is_empty() {
                continue;
            }
            let fields = split_csv_line_basic(&line);
            let mut idx_tuple = Vec::with_capacity(qis.len());
            let mut valid = true;
            for i in 0..qis.len() {
                if col_idx[i] >= fields.len() {
                    valid = false;
                    break;
                }
                let raw = fields[col_idx[i]].trim();
                if qis[i].is_categorical {
                    let Some(idx) = categorical_domains[i].iter().position(|v| v == raw).map(|v| v as i64) else {
                        valid = false;
                        break;
                    };
                    idx_tuple.push(idx);
                } else {
                    let Ok(v) = raw.parse::<f64>() else {
                        valid = false;
                        break;
                    };
                    let mn = qis[i].min_value.unwrap_or(0.0);
                    let step = initial_ri[i].max(1) as f64;
                    let idx = ((v - mn) / step).floor().max(0.0) as i64;
                    idx_tuple.push(idx);
                }
            }
            if !valid {
                continue;
            }
            *hist.entry(idx_tuple).or_insert(0) += 1;
            total_records += 1;
        }
    }

    if hist.is_empty() || total_records <= 0 {
        return Err(validation(
            "GENERALIZATION_FAILED",
            "No valid histogram records produced",
            "check QI columns",
        ));
    }

    Ok((hist, total_records))
}

// ── Histogram merging and scoring ─────────────────────────────────────────────

/// Merges a fine-grained sparse histogram to the coarser granularity of `node`.
///
/// Each key `k` is mapped to `k[i] / node[i]` (integer division), and counts
/// from keys that map to the same coarse key are summed.
///
/// # Arguments
/// * `hist` — fine-grained histogram (built at `initial_ri` resolution).
/// * `node` — coarser bin widths (one per QI, ≥ 1).
fn merge_sparse_equivalence(hist: &SparseHist, node: &[i64]) -> SparseHist {
    let mut merged: SparseHist = BTreeMap::new();
    for (idx, count) in hist {
        let mut m = Vec::with_capacity(idx.len());
        for (i, v) in idx.iter().enumerate() {
            m.push(*v / node[i].max(1));
        }
        *merged.entry(m).or_insert(0) += *count;
    }
    merged
}

/// Counts the total number of records in equivalence classes smaller than `k`
/// (i.e., records that would be suppressed at this node).
///
/// # Arguments
/// * `hist` — merged histogram at node granularity.
/// * `k` — the k-anonymity threshold.
fn suppression_count_sparse(hist: &SparseHist, k: i64) -> i64 {
    hist.values().filter(|&&v| v > 0 && v < k).sum()
}

/// Computes the DM* (Discernibility Metric) and equivalence class count for a
/// merged histogram.
///
/// Only equivalence classes with count ≥ k contribute to DM*; smaller classes
/// are excluded (they will be suppressed).
///
/// Returns `(dm_star, num_equivalence_classes)`.
fn compute_dm_star_sparse(hist: &SparseHist, k: i64) -> (i64, i64) {
    let mut dm = 0_i64;
    let mut eq = 0_i64;
    for &v in hist.values() {
        if v >= k {
            dm += v * v;
            eq += 1;
        }
    }
    (dm, eq)
}

// ── Lattice pass/fail propagation ─────────────────────────────────────────────

/// Mark every unmarked descendant of `node` (all dims ≥ node) as passing.
/// Only levels ≥ `node_level` are scanned — descendants cannot live at lower levels.
/// Replaces the old O(N²) BFS with a single O(N) linear scan over the relevant slice.
fn mark_subtree_pass(
    tree: &[Vec<Vec<i64>>],
    node_status: &mut BTreeMap<Vec<i64>, Option<bool>>,
    node: &[i64],
    node_level: usize,
) {
    for level in &tree[node_level..] {
        for candidate in level {
            if node_status.get(candidate).copied().flatten().is_none()
                && candidate.iter().zip(node.iter()).all(|(c, p)| c >= p)
            {
                node_status.insert(candidate.clone(), Some(true));
            }
        }
    }
}

/// Mark every unmarked ancestor of `node` (all dims ≤ node) as failing.
/// Only levels ≤ `node_level` are scanned — ancestors cannot live at higher levels.
/// Replaces the old O(N²) BFS with a single O(N) linear scan over the relevant slice.
fn mark_parents_fail(
    tree: &[Vec<Vec<i64>>],
    node_status: &mut BTreeMap<Vec<i64>, Option<bool>>,
    node: &[i64],
    node_level: usize,
) {
    let upper = (node_level + 1).min(tree.len());
    for level in &tree[..upper] {
        for candidate in level {
            if node_status.get(candidate).copied().flatten().is_none()
                && candidate.iter().zip(node.iter()).all(|(p, c)| p <= c)
            {
                node_status.insert(candidate.clone(), Some(false));
            }
        }
    }
}

// ── Node scoring ──────────────────────────────────────────────────────────────

/// Evaluates and ranks up to 5 of the best passing nodes by `(dm_star,
/// suppression_count, -num_equivalence_classes, node)` ascending.
///
/// # Arguments
/// * `base_hist` — fine-grained histogram at `initial_ri` resolution.
/// * `pass_nodes` — all nodes that passed the k-anonymity + suppression check.
/// * `k` — k-anonymity threshold.
///
/// # Returns
/// At most 5 [`Ola2NodeScore`] entries sorted from best to worst.
fn compute_top_ola2_nodes(
    base_hist: &SparseHist,
    pass_nodes: &[Vec<i64>],
    k: i64,
) -> Vec<Ola2NodeScore> {
    let mut out = Vec::new();
    for node in pass_nodes {
        let merged = merge_sparse_equivalence(base_hist, node);
        let (dm, eq) = compute_dm_star_sparse(&merged, k);
        let suppression_count = suppression_count_sparse(&merged, k);
        out.push(Ola2NodeScore {
            node: node.clone(),
            dm_star: dm,
            num_equivalence_classes: eq,
            suppression_count,
        });
    }
    out.sort_by(|a, b| {
        (a.dm_star, a.suppression_count, -a.num_equivalence_classes, a.node.clone()).cmp(&(
            b.dm_star,
            b.suppression_count,
            -b.num_equivalence_classes,
            b.node.clone(),
        ))
    });
    out.into_iter().take(5).collect()
}

// ── OLA-2 public interface ────────────────────────────────────────────────────

/// Convenience wrapper around [`find_ola2_best_rf_detailed`] that returns only
/// the winning generalization factors and summary scores.
///
/// # Returns
/// `(best_rf, lowest_dm_star, num_equivalence_classes)`
///
/// # Errors
/// Propagates all errors from [`find_ola2_best_rf_detailed`].
pub fn find_ola2_best_rf(
    qis: &[QuasiIdentifierLite],
    base_hist: &SparseHist,
    initial_ri: &[i64],
    size_factors: &BTreeMap<String, i64>,
    k: i64,
    suppression_limit: f64,
    total_records: i64,
) -> Result<(Vec<i64>, i64, i64), PipelineError> {
    let detailed = find_ola2_best_rf_detailed(
        qis,
        base_hist,
        initial_ri,
        size_factors,
        k,
        suppression_limit,
        total_records,
    )?;
    Ok((
        detailed.best_rf,
        detailed.lowest_dm_star,
        detailed.num_equivalence_classes,
    ))
}

/// Runs the full OLA-2 binary lattice search and returns detailed results.
///
/// The algorithm:
/// 1. Build the generalization lattice from `initial_ri`.
/// 2. Maintain a `node_status` map (unmarked / pass / fail).
/// 3. Repeat until all nodes are marked:
///    a. Pick the median unmarked level, then the median unmarked node on it.
///    b. Merge the histogram to that node's granularity and count suppressed records.
///    c. If the suppression fraction is within `suppression_limit`, mark the node
///       **pass** and propagate pass to all descendants.
///    d. Otherwise mark it **fail** and propagate fail to all ancestors.
/// 4. Score all passing nodes with DM* and return the top-5.
///
/// # Arguments
/// * `qis` — quasi-identifier descriptors.
/// * `base_hist` — sparse histogram at `initial_ri` granularity.
/// * `initial_ri` — starting node from OLA-1.
/// * `size_factors` — multiplication factors keyed by base column name.
/// * `k` — k-anonymity threshold (must be > 0).
/// * `suppression_limit` — maximum fraction of records that may be suppressed (`[0.0, 1.0]`).
/// * `total_records` — total valid records in `base_hist` (must be > 0).
///
/// # Returns
/// An [`Ola2SearchResult`] containing `best_rf`, `lowest_dm_star`,
/// `num_equivalence_classes`, and the top-5 `top_nodes`.
///
/// # Errors
/// - `GENERALIZATION_FAILED` — invalid inputs or no feasible node found.
pub fn find_ola2_best_rf_detailed(
    qis: &[QuasiIdentifierLite],
    base_hist: &SparseHist,
    initial_ri: &[i64],
    size_factors: &BTreeMap<String, i64>,
    k: i64,
    suppression_limit: f64,
    total_records: i64,
) -> Result<Ola2SearchResult, PipelineError> {
    if k <= 0 {
        return Err(validation("GENERALIZATION_FAILED", "k must be > 0", "OLA-2"));
    }
    if !(0.0..=1.0).contains(&suppression_limit) {
        return Err(validation(
            "GENERALIZATION_FAILED",
            "suppression_limit must be in [0,1]",
            "OLA-2",
        ));
    }
    if total_records <= 0 {
        return Err(validation("GENERALIZATION_FAILED", "total_records must be > 0", "OLA-2"));
    }
    if initial_ri.len() != qis.len() {
        return Err(validation("GENERALIZATION_FAILED", "initial_ri length mismatch", "OLA-2"));
    }
    let allowed = suppression_limit * total_records as f64;
    let tree = build_tree_levels(qis, initial_ri, size_factors, false)?;
    let mut node_status: BTreeMap<Vec<i64>, Option<bool>> = BTreeMap::new();
    // Pre-build node → level index map so mark functions can restrict their scan
    let mut node_to_level: BTreeMap<Vec<i64>, usize> = BTreeMap::new();
    for (level_idx, level) in tree.iter().enumerate() {
        for node in level {
            node_status.entry(node.clone()).or_insert(None);
            node_to_level.insert(node.clone(), level_idx);
        }
    }

    let mut pass_nodes: Vec<Vec<i64>> = Vec::new();
    while node_status.values().any(Option::is_none) {
        let unmarked_levels: Vec<usize> = tree
            .iter()
            .enumerate()
            .filter_map(|(i, level)| {
                if level.iter().any(|n| node_status.get(n).copied().flatten().is_none()) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        if unmarked_levels.is_empty() {
            break;
        }
        let mid_level = unmarked_levels[unmarked_levels.len() / 2];
        let mut sorted_nodes: Vec<Vec<i64>> = tree[mid_level]
            .iter()
            .filter(|n| node_status.get(*n).copied().flatten().is_none())
            .cloned()
            .collect();
        sorted_nodes.sort_by(|a, b| b.cmp(a));
        if sorted_nodes.is_empty() {
            continue;
        }
        let node = sorted_nodes[sorted_nodes.len() / 2].clone();
        if node_status.get(&node).copied().flatten().is_some() {
            continue;
        }
        let node_level = node_to_level.get(&node).copied().unwrap_or(mid_level);
        let merged = merge_sparse_equivalence(base_hist, &node);
        let suppression_count = suppression_count_sparse(&merged, k);
        let passes_k = suppression_count == 0;
        if passes_k || (suppression_count as f64) <= allowed {
            node_status.insert(node.clone(), Some(true));
            pass_nodes.push(node.clone());
            mark_subtree_pass(&tree, &mut node_status, &node, node_level);
        } else {
            node_status.insert(node.clone(), Some(false));
            mark_parents_fail(&tree, &mut node_status, &node, node_level);
        }
    }

    if pass_nodes.is_empty() {
        return Err(validation(
            "GENERALIZATION_FAILED",
            "No node satisfies k-anonymity constraints",
            "OLA-2",
        ));
    }

    let scored = compute_top_ola2_nodes(base_hist, &pass_nodes, k);
    let best = scored
        .first()
        .ok_or_else(|| validation("GENERALIZATION_FAILED", "No valid OLA-2 node scores", "OLA-2"))?;
    Ok(Ola2SearchResult {
        best_rf: best.node.clone(),
        lowest_dm_star: best.dm_star,
        num_equivalence_classes: best.num_equivalence_classes,
        top_nodes: scored,
    })
}

// ── Post-processing statistics ────────────────────────────────────────────────

/// Computes a frequency distribution of equivalence-class sizes at the chosen
/// generalization node.
///
/// Only classes with count ≥ k are included (suppressed classes are excluded).
/// The returned map can be serialized to JSON for audit logs.
///
/// # Arguments
/// * `hist` — fine-grained histogram at `initial_ri` resolution.
/// * `rf` — chosen generalization factors (the `best_rf` from OLA-2).
/// * `k` — k-anonymity threshold.
///
/// # Returns
/// Map from equivalence-class size to the number of classes with that size.
pub fn equivalence_class_stats(hist: &SparseHist, rf: &[i64], k: i64) -> BTreeMap<i64, i64> {
    let merged = merge_sparse_equivalence(hist, rf);
    let mut stats = BTreeMap::new();
    for v in merged.values() {
        if *v >= k {
            *stats.entry(*v).or_insert(0) += 1;
        }
    }
    stats
}
