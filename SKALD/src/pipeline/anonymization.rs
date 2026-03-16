use crate::pipeline::bootstrap::{split_csv_line_basic, validation, PipelineError, RuntimeConfig};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
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
    let mut dynamic_min_max: BTreeMap<String, (f64, f64)> = BTreeMap::new();

    for col_name in numerical_columns {
        let mut min_val: Option<f64> = None;
        let mut max_val: Option<f64> = None;

        for chunk_path in chunk_paths {
            let f = fs::File::open(chunk_path)?;
            let r = BufReader::new(f);
            let mut lines = r.lines();
            let header = lines
                .next()
                .ok_or_else(|| validation("ENCODING_FAILED", "Chunk missing header", &chunk_path.display().to_string()))?
                .map_err(|e| validation("ENCODING_FAILED", "Failed reading header", &e.to_string()))?;
            let headers = split_csv_line_basic(&header);
            let col_idx = headers
                .iter()
                .position(|h| h == col_name)
                .ok_or_else(|| validation("GENERALIZATION_FAILED", "Numerical column not found", col_name))?;

            for line in lines {
                let line = line.map_err(|e| validation("GENERALIZATION_FAILED", "Failed reading row", &e.to_string()))?;
                let fields = split_csv_line_basic(&line);
                if col_idx >= fields.len() {
                    continue;
                }
                if let Ok(v) = fields[col_idx].trim().parse::<f64>() {
                    min_val = Some(min_val.map_or(v, |m| m.min(v)));
                    max_val = Some(max_val.map_or(v, |m| m.max(v)));
                }
            }
        }

        let mn = min_val.unwrap_or(0.0);
        let mx = max_val.unwrap_or(0.0);
        dynamic_min_max.insert(col_name.clone(), (mn, mx));
    }

    Ok(dynamic_min_max)
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
        return Err(validation("GENERALIZATION_FAILED", "No QIs for OLA-1", "qis"));
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

    while node_status.values().any(Option::is_none) {
        for level in &tree {
            for node in level {
                if node_status.get(node).and_then(|v| *v).is_some()
                    || node_status.get(node) == Some(&Some(false))
                {
                    continue;
                }
                let classes = calculate_equivalence_classes_ola1(qis, node)?;
                if classes <= max_eq {
                    node_status.insert(node.clone(), Some(true));
                } else {
                    // Mirrors Python behavior where only current node is reliably marked fail.
                    node_status.insert(node.clone(), Some(false));
                }
            }
        }
    }

    let mut passing: Vec<Vec<i64>> = node_status
        .into_iter()
        .filter_map(|(n, st)| if st == Some(true) { Some(n) } else { None })
        .collect();
    passing.sort();
    passing.into_iter().next().ok_or_else(|| {
        validation(
            "GENERALIZATION_FAILED",
            "No generalization satisfies equivalence class constraint",
            "OLA-1",
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

    let mut hist: SparseHist = BTreeMap::new();
    let mut total_records: i64 = 0;
    let mut categorical_domains: Vec<Vec<String>> = vec![Vec::new(); qis.len()];
    let mut domains_built = false;

    for chunk in chunk_paths {
        let f = fs::File::open(chunk)?;
        let r = BufReader::new(f);
        let mut lines = r.lines();
        let header = lines
            .next()
            .ok_or_else(|| validation("GENERALIZATION_FAILED", "chunk missing header", &chunk.display().to_string()))?
            .map_err(|e| validation("GENERALIZATION_FAILED", "failed reading chunk header", &e.to_string()))?;
        let headers = split_csv_line_basic(&header);

        let mut col_idx = Vec::with_capacity(qis.len());
        for qi in qis {
            let fallback = base_col_name(&qi.column_name);
            let idx = headers
                .iter()
                .position(|h| h == &qi.column_name)
                .or_else(|| headers.iter().position(|h| h == &fallback))
                .ok_or_else(|| validation("GENERALIZATION_FAILED", "QI column missing in chunk", &qi.column_name))?;
            col_idx.push(idx);
        }

        if !domains_built {
            let mut cat_sets: Vec<BTreeSet<String>> = vec![BTreeSet::new(); qis.len()];
            for line in lines.by_ref() {
                let line = line.map_err(|e| validation("GENERALIZATION_FAILED", "failed reading chunk row", &e.to_string()))?;
                if line.trim().is_empty() {
                    continue;
                }
                let fields = split_csv_line_basic(&line);
                for i in 0..qis.len() {
                    if !qis[i].is_categorical || col_idx[i] >= fields.len() {
                        continue;
                    }
                    cat_sets[i].insert(fields[col_idx[i]].trim().to_string());
                }
            }
            for i in 0..qis.len() {
                if qis[i].is_categorical {
                    categorical_domains[i] = cat_sets[i].iter().cloned().collect();
                }
            }
            domains_built = true;

            let f2 = fs::File::open(chunk)?;
            let r2 = BufReader::new(f2);
            lines = r2.lines();
            let _ = lines.next();
        }

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

fn mark_subtree_pass(
    tree: &[Vec<Vec<i64>>],
    node_status: &mut BTreeMap<Vec<i64>, Option<bool>>,
    node: &[i64],
) {
    let mut q = VecDeque::new();
    q.push_back(node.to_vec());
    while let Some(current) = q.pop_front() {
        if node_status.get(&current).and_then(|v| *v).is_none() {
            node_status.insert(current.clone(), Some(true));
        }
        for level in tree {
            for child in level {
                if child.iter().zip(current.iter()).all(|(c, p)| c >= p)
                    && node_status.get(child).copied().flatten().is_none()
                {
                    node_status.insert(child.clone(), Some(true));
                    q.push_back(child.clone());
                }
            }
        }
    }
}

fn mark_parents_fail(
    tree: &[Vec<Vec<i64>>],
    node_status: &mut BTreeMap<Vec<i64>, Option<bool>>,
    node: &[i64],
) {
    let mut q = VecDeque::new();
    q.push_back(node.to_vec());
    while let Some(current) = q.pop_front() {
        if node_status.get(&current).copied().flatten().is_none() {
            node_status.insert(current.clone(), Some(false));
        }
        for level in tree.iter().rev() {
            for parent in level {
                if parent.iter().zip(current.iter()).all(|(p, c)| p <= c)
                    && node_status.get(parent).copied().flatten().is_none()
                {
                    node_status.insert(parent.clone(), Some(false));
                    q.push_back(parent.clone());
                }
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
    for level in &tree {
        for node in level {
            node_status.entry(node.clone()).or_insert(None);
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
        let merged = merge_sparse_equivalence(base_hist, &node);
        let suppression_count = suppression_count_sparse(&merged, k);
        let passes_k = suppression_count == 0;
        if passes_k || (suppression_count as f64) <= allowed {
            node_status.insert(node.clone(), Some(true));
            pass_nodes.push(node.clone());
            mark_subtree_pass(&tree, &mut node_status, &node);
        } else {
            node_status.insert(node.clone(), Some(false));
            mark_parents_fail(&tree, &mut node_status, &node);
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

    let mut output_chunk_paths = Vec::new();
    for (idx, chunk) in chunk_paths.iter().enumerate() {
        let f = fs::File::open(chunk)?;
        let r = BufReader::new(f);
        let mut lines = r.lines();
        let header = lines
            .next()
            .ok_or_else(|| validation("GENERALIZATION_FAILED", "chunk missing header", &chunk.display().to_string()))?
            .map_err(|e| validation("GENERALIZATION_FAILED", "failed reading chunk header", &e.to_string()))?;
        let headers = split_csv_line_basic(&header);
        let mut rows: Vec<Vec<String>> = Vec::new();
        for line in lines {
            let line = line.map_err(|e| validation("GENERALIZATION_FAILED", "failed reading chunk row", &e.to_string()))?;
            rows.push(split_csv_line_basic(&line));
        }

        // Generalize QIs in place on base columns.
        for (qi, bw) in qis.iter().zip(final_rf.iter()) {
            if qi.is_categorical {
                let Some(src_idx) = headers.iter().position(|h| h == &qi.column_name) else { continue; };
                for row in &mut rows {
                    if let Some(v) = row.get(src_idx).cloned() {
                        row[src_idx] = generalize_categorical_value(&qi.column_name, &v, *bw);
                    }
                }
                continue;
            }
            let src_col = base_col_name(&qi.column_name);
            let Some(src_idx) = headers.iter().position(|h| h == &src_col) else { continue; };

            let valid_vals: Vec<i64> = rows
                .iter()
                .filter_map(|r| r.get(src_idx))
                .filter_map(|s| s.trim().parse::<f64>().ok())
                .map(|v| v.round() as i64)
                .collect();
            if valid_vals.is_empty() {
                continue;
            }
            let min_val = *valid_vals.iter().min().unwrap_or(&0);

            for row in &mut rows {
                if let Some(s) = row.get(src_idx).cloned() {
                    if let Ok(v) = s.trim().parse::<f64>() {
                        row[src_idx] = generalize_numeric_label(v.round() as i64, min_val, (*bw).max(1));
                    }
                }
            }
        }

        // Suppress (mark '*') QI-only columns for classes < k.
        let mut qi_output_idx: Vec<usize> = Vec::with_capacity(qis.len());
        for qi in qis {
            let out_col = if qi.is_categorical {
                qi.column_name.clone()
            } else {
                base_col_name(&qi.column_name)
            };
            if let Some(i) = headers.iter().position(|h| h == &out_col) {
                qi_output_idx.push(i);
            }
        }

        if !qi_output_idx.is_empty() && k > 1 {
            let mut class_counts: BTreeMap<Vec<String>, i64> = BTreeMap::new();
            for row in &rows {
                let key: Vec<String> = qi_output_idx
                    .iter()
                    .map(|&i| row.get(i).cloned().unwrap_or_default())
                    .collect();
                *class_counts.entry(key).or_insert(0) += 1;
            }
            for row in &mut rows {
                let key: Vec<String> = qi_output_idx
                    .iter()
                    .map(|&i| row.get(i).cloned().unwrap_or_default())
                    .collect();
                if class_counts.get(&key).copied().unwrap_or(0) < k {
                    for &i in &qi_output_idx {
                        if i < row.len() {
                            row[i] = "*".to_string();
                        }
                    }
                }
            }
        }

        let out_headers = headers.clone();
        let out_rows = rows;

        let out_name = generalized_chunk_name(output_path, idx + 1);
        let out_path = output_dir.join(out_name);
        let mut w = BufWriter::new(fs::File::create(&out_path)?);
        w.write_all(out_headers.join(",").as_bytes())?;
        w.write_all(b"\n")?;
        for row in out_rows {
            w.write_all(row.join(",").as_bytes())?;
            w.write_all(b"\n")?;
        }
        w.flush()?;
        output_chunk_paths.push(out_path);
    }

    // combine chunk outputs
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
            let line = line.map_err(|e| validation("GENERALIZATION_FAILED", "failed reading generalized chunk", &e.to_string()))?;
            if i > 0 && line_no == 0 {
                continue;
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
