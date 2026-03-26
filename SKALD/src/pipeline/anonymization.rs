use crate::pipeline::bootstrap::{split_csv_line_basic, validation, PipelineError, RuntimeConfig};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct QuasiIdentifierLite {
    pub column_name: String,
    pub is_categorical: bool,
    pub min_value: Option<f64>,
    pub max_value: Option<f64>,
}

impl QuasiIdentifierLite {
    fn get_range(&self) -> f64 {
        match (self.min_value, self.max_value) {
            (Some(a), Some(b)) => (b - a + 1.0).max(1.0),
            _ => 1.0,
        }
    }
}

pub type SparseHist = BTreeMap<Vec<i64>, i64>;

#[derive(Debug, Clone, Serialize)]
pub struct Ola2NodeScore {
    pub node: Vec<i64>,
    pub dm_star: i64,
    pub num_equivalence_classes: i64,
    pub suppression_count: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Ola2SearchResult {
    pub best_rf: Vec<i64>,
    pub lowest_dm_star: i64,
    pub num_equivalence_classes: i64,
    pub top_nodes: Vec<Ola2NodeScore>,
}

fn ceil_div_i64(a: i64, b: i64) -> i64 {
    let aa = a.max(0);
    let bb = b.max(1);
    (aa + bb - 1) / bb
}

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

fn max_categorical_level_ola2(col: &str) -> i64 {
    match col.trim().to_lowercase().as_str() {
        "blood group" => 3,
        "gender" => 2,
        _ => 4,
    }
}

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

pub fn base_col_name(col: &str) -> String {
    if let Some(v) = col.strip_suffix("_scaled_encoded") {
        v.to_string()
    } else if let Some(v) = col.strip_suffix("_encoded") {
        v.to_string()
    } else if let Some(v) = col.strip_suffix("_scaled") {
        v.to_string()
    } else {
        col.to_string()
    }
}

pub fn compute_numerical_min_max(
    chunk_paths: &[PathBuf],
    numerical_columns: &[String],
) -> Result<BTreeMap<String, (f64, f64)>, PipelineError> {
    // (min, max) accumulators — one entry per column, updated in a single pass per chunk
    let mut acc: BTreeMap<String, (Option<f64>, Option<f64>)> =
        numerical_columns.iter().map(|c| (c.clone(), (None, None))).collect();

    for chunk_path in chunk_paths {
        let f = fs::File::open(chunk_path)?;
        let r = BufReader::new(f);
        let mut lines = r.lines();
        let header = lines
            .next()
            .ok_or_else(|| validation("ENCODING_FAILED", "Chunk missing header", &chunk_path.display().to_string()))?
            .map_err(|e| validation("ENCODING_FAILED", "Failed reading header", &e.to_string()))?;
        let headers = split_csv_line_basic(&header);

        // Resolve column indices once per chunk (same schema across all chunks)
        let col_indices: Vec<Option<usize>> = numerical_columns
            .iter()
            .map(|col| headers.iter().position(|h| h == col))
            .collect();

        for line in lines {
            let line = line.map_err(|e| validation("GENERALIZATION_FAILED", "Failed reading row", &e.to_string()))?;
            if line.trim().is_empty() {
                continue;
            }
            let fields = split_csv_line_basic(&line);
            for (col_name, col_idx_opt) in numerical_columns.iter().zip(col_indices.iter()) {
                let Some(col_idx) = col_idx_opt else { continue };
                if *col_idx >= fields.len() {
                    continue;
                }
                if let Ok(v) = fields[*col_idx].trim().parse::<f64>() {
                    let entry = acc.get_mut(col_name).unwrap();
                    entry.0 = Some(entry.0.map_or(v, |m: f64| m.min(v)));
                    entry.1 = Some(entry.1.map_or(v, |m: f64| m.max(v)));
                }
            }
        }
    }

    Ok(acc
        .into_iter()
        .map(|(k, (mn, mx))| (k, (mn.unwrap_or(0.0), mx.unwrap_or(0.0))))
        .collect())
}

pub fn build_quasi_identifiers(
    cfg: &RuntimeConfig,
    dynamic_min_max: &BTreeMap<String, (f64, f64)>,
) -> Result<Vec<QuasiIdentifierLite>, PipelineError> {
    let mut out = Vec::new();

    for qi in &cfg.numerical_qis {
        let (mn, mx) = dynamic_min_max.get(&qi.column).copied().unwrap_or((0.0, 0.0));
        out.push(QuasiIdentifierLite {
            column_name: qi.column.clone(),
            is_categorical: false,
            min_value: Some(mn),
            max_value: Some(mx),
        });
    }

    for c in &cfg.categorical_qis {
        out.push(QuasiIdentifierLite {
            column_name: c.clone(),
            is_categorical: true,
            min_value: None,
            max_value: None,
        });
    }

    if out.is_empty() {
        return Err(validation("CONFIG_INVALID", "No quasi-identifiers defined", "quasi_identifiers"));
    }

    Ok(out)
}

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

fn suppression_count_sparse(hist: &SparseHist, k: i64) -> i64 {
    hist.values().filter(|&&v| v > 0 && v < k).sum()
}

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

fn generalize_numeric_label(value: i64, min_val: i64, step: i64) -> String {
    let bucket_start = min_val + ((value - min_val) / step.max(1)) * step.max(1);
    let bucket_end = bucket_start + step.max(1) - 1;
    format!("[{}-{}]", bucket_start, bucket_end)
}

fn generalize_categorical_value(column_name: &str, value: &str, level: i64) -> String {
    let key = column_name.trim().to_lowercase();
    let lvl = level.max(1);
    match key.as_str() {
        "blood group" => {
            if lvl <= 1 {
                value.to_string()
            } else if lvl == 2 {
                match value {
                    "A+" | "A-" => "A".to_string(),
                    "B+" | "B-" => "B".to_string(),
                    "AB+" | "AB-" => "AB".to_string(),
                    "O+" | "O-" => "O".to_string(),
                    _ => "Other".to_string(),
                }
            } else {
                "*".to_string()
            }
        }
        "gender" => {
            if lvl <= 1 {
                value.to_string()
            } else {
                "*".to_string()
            }
        }
        "profession" => {
            if lvl <= 1 {
                value.to_string()
            } else if lvl == 2 {
                match value {
                    "Medical Specialists" | "Allied Health" | "Nursing" | "Healthcare Support" => "Healthcare".to_string(),
                    "K-12 Education Teacher" | "Higher Education Teacher" | "Supplemental Education Teacher" | "University Professor" => "Education".to_string(),
                    "Performing Arts" | "Visual & Media Arts" | "Design" | "Mixed Media Artist" => "Creative".to_string(),
                    "Traditional Engineering" | "Software Engineering" | "Data & Analytics" | "AI & Machine Learning" => "Engineering".to_string(),
                    _ => "Other".to_string(),
                }
            } else if lvl == 3 {
                match value {
                    "Medical Specialists"
                    | "Allied Health"
                    | "Nursing"
                    | "Healthcare Support"
                    | "K-12 Education Teacher"
                    | "Higher Education Teacher"
                    | "Supplemental Education Teacher"
                    | "University Professor" => "Service Sector".to_string(),
                    "Performing Arts"
                    | "Visual & Media Arts"
                    | "Design"
                    | "Mixed Media Artist"
                    | "Traditional Engineering"
                    | "Software Engineering"
                    | "Data & Analytics"
                    | "AI & Machine Learning" => "Non-Service".to_string(),
                    _ => "Other".to_string(),
                }
            } else {
                "*".to_string()
            }
        }
        _ => value.to_string(),
    }
}

fn generalized_chunk_name(output_path: &str, idx: usize) -> String {
    let base = output_path.trim_end_matches(".csv");
    format!("{base}_chunk{idx}.csv")
}

pub fn generalize_and_write_outputs(
    chunk_paths: &[PathBuf],
    qis: &[QuasiIdentifierLite],
    final_rf: &[i64],
    k: i64,
    output_dir: &Path,
    output_path: &str,
) -> Result<(), PipelineError> {
    if final_rf.len() != qis.len() {
        return Err(validation("GENERALIZATION_FAILED", "final_rf length mismatch", "generalization"));
    }
    fs::create_dir_all(output_dir)?;

    // Resolve column layout from first chunk header (uniform schema across all chunks)
    let (headers, qi_src_indices, qi_output_idx) = {
        let first = chunk_paths
            .first()
            .ok_or_else(|| validation("GENERALIZATION_FAILED", "no chunks provided", "generalization"))?;
        let f = fs::File::open(first)?;
        let mut lines = BufReader::new(f).lines();
        let header = lines
            .next()
            .ok_or_else(|| validation("GENERALIZATION_FAILED", "chunk missing header", &first.display().to_string()))?
            .map_err(|e| validation("GENERALIZATION_FAILED", "failed reading header", &e.to_string()))?;
        let headers = split_csv_line_basic(&header);

        // Source column index for each QI (may differ from output index for numerical)
        let qi_src: Vec<Option<usize>> = qis
            .iter()
            .map(|qi| {
                if qi.is_categorical {
                    headers.iter().position(|h| h == &qi.column_name)
                } else {
                    let src = base_col_name(&qi.column_name);
                    headers.iter().position(|h| h == &src)
                }
            })
            .collect();

        // Output QI column indices (used for EC suppression key)
        let qi_out: Vec<usize> = qis
            .iter()
            .filter_map(|qi| {
                let col = if qi.is_categorical { qi.column_name.clone() } else { base_col_name(&qi.column_name) };
                headers.iter().position(|h| h == &col)
            })
            .collect();

        (headers, qi_src, qi_out)
    };

    // Use global min from qi.min_value (computed across all chunks by compute_numerical_min_max)
    let global_mins: Vec<i64> = qis.iter().map(|qi| qi.min_value.unwrap_or(0.0).floor() as i64).collect();

    let mut output_chunk_paths = Vec::new();

    for (chunk_idx, chunk) in chunk_paths.iter().enumerate() {
        // --- Pass 1: generalize each row, write to temp file, accumulate EC counts ---
        let tmp_path = output_dir.join(format!("_gen_tmp_{chunk_idx}.csv"));
        let mut class_counts: BTreeMap<Vec<String>, i64> = BTreeMap::new();

        {
            let f = fs::File::open(chunk)?;
            let mut lines = BufReader::new(f).lines();
            let _ = lines.next(); // skip header

            let mut w = BufWriter::new(fs::File::create(&tmp_path)?);
            w.write_all(headers.join(",").as_bytes())?;
            w.write_all(b"\n")?;

            for line in lines {
                let line = line.map_err(|e| {
                    validation("GENERALIZATION_FAILED", "failed reading chunk row", &e.to_string())
                })?;
                if line.trim().is_empty() {
                    continue;
                }
                let mut fields = split_csv_line_basic(&line);

                // Apply generalization in-place
                for ((qi, bw), src_idx_opt) in qis.iter().zip(final_rf.iter()).zip(qi_src_indices.iter()) {
                    let Some(src_idx) = src_idx_opt else { continue };
                    if *src_idx >= fields.len() {
                        continue;
                    }
                    if qi.is_categorical {
                        let v = fields[*src_idx].clone();
                        fields[*src_idx] = generalize_categorical_value(&qi.column_name, &v, *bw);
                    } else {
                        if let Ok(v) = fields[*src_idx].trim().parse::<f64>() {
                            let qi_pos = qis.iter().position(|q| std::ptr::eq(q, qi)).unwrap_or(0);
                            fields[*src_idx] =
                                generalize_numeric_label(v.round() as i64, global_mins[qi_pos], (*bw).max(1));
                        }
                    }
                }

                // Accumulate EC key (generalized QI values only)
                if !qi_output_idx.is_empty() && k > 1 {
                    let key: Vec<String> =
                        qi_output_idx.iter().map(|&i| fields.get(i).cloned().unwrap_or_default()).collect();
                    *class_counts.entry(key).or_insert(0) += 1;
                }

                w.write_all(fields.join(",").as_bytes())?;
                w.write_all(b"\n")?;
            }
            w.flush()?;
        }

        // --- Pass 2: apply suppression using EC counts, write final chunk output ---
        let out_name = generalized_chunk_name(output_path, chunk_idx + 1);
        let out_path = output_dir.join(out_name);
        {
            let f = fs::File::open(&tmp_path)?;
            let mut lines = BufReader::new(f).lines();
            let _ = lines.next(); // skip header written by pass 1

            let mut w = BufWriter::new(fs::File::create(&out_path)?);
            w.write_all(headers.join(",").as_bytes())?;
            w.write_all(b"\n")?;

            for line in lines {
                let line = line.map_err(|e| {
                    validation("GENERALIZATION_FAILED", "failed reading temp chunk", &e.to_string())
                })?;
                if line.trim().is_empty() {
                    continue;
                }
                let mut fields = split_csv_line_basic(&line);

                if !qi_output_idx.is_empty() && k > 1 {
                    let key: Vec<String> =
                        qi_output_idx.iter().map(|&i| fields.get(i).cloned().unwrap_or_default()).collect();
                    if class_counts.get(&key).copied().unwrap_or(0) < k {
                        for &i in &qi_output_idx {
                            if i < fields.len() {
                                fields[i] = "*".to_string();
                            }
                        }
                    }
                }

                w.write_all(fields.join(",").as_bytes())?;
                w.write_all(b"\n")?;
            }
            w.flush()?;
        }

        let _ = fs::remove_file(&tmp_path);
        output_chunk_paths.push(out_path);
    }

    // Merge chunk outputs into single final file
    let final_path = if Path::new(output_path).is_absolute() {
        PathBuf::from(output_path)
    } else {
        output_dir.join(output_path)
    };
    let mut final_writer = BufWriter::new(fs::File::create(&final_path)?);
    for (i, p) in output_chunk_paths.iter().enumerate() {
        let f = fs::File::open(p)?;
        let r = BufReader::new(f);
        for (line_no, line) in r.lines().enumerate() {
            let line = line.map_err(|e| {
                validation("GENERALIZATION_FAILED", "failed reading generalized chunk", &e.to_string())
            })?;
            if i > 0 && line_no == 0 {
                continue; // skip repeated header
            }
            final_writer.write_all(line.as_bytes())?;
            final_writer.write_all(b"\n")?;
        }
    }
    final_writer.flush()?;
    for p in output_chunk_paths {
        let _ = fs::remove_file(p);
    }

    Ok(())
}
