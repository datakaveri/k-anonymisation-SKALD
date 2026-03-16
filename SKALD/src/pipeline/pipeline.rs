use crate::pipeline::anonymization::{
    build_quasi_identifiers, build_sparse_histogram, compute_numerical_min_max, equivalence_class_stats,
    find_ola1_initial_ri, find_ola2_best_rf_detailed, generalize_and_write_outputs,
};
use crate::pipeline::bootstrap::{
    ensure_output_dir, find_first_json_config, list_non_empty_csvs, parse_runtime_config, split_csv_by_ram,
    PipelineError, StatusPayload,
};
use crate::pipeline::preprocess::preprocess_chunks;
use serde_json::json;
use std::fs;
use std::path::Path;

pub fn run_pipeline(root: &Path) -> Result<StatusPayload, PipelineError> {
    let config_path = find_first_json_config(&root.join("config"))?;
    let cfg = parse_runtime_config(&config_path)?;

    let csvs = list_non_empty_csvs(&root.join("data"))?;
    let (chunk_paths, rows_per_chunk) = split_csv_by_ram(&root.join("data"), &root.join("chunks"), None, None)?;

    preprocess_chunks(&chunk_paths, &cfg)?;

    let numerical_cols = cfg.numerical_qis.iter().map(|q| q.column.clone()).collect::<Vec<_>>();
    let dynamic_min_max = compute_numerical_min_max(&chunk_paths, &numerical_cols)?;
    let qis = build_quasi_identifiers(&cfg, &dynamic_min_max)?;

    let max_eq = estimate_available_ram_bytes().unwrap_or(32_000_000) / 32;
    let initial_ri = find_ola1_initial_ri(&qis, chunk_paths.len() as i64, max_eq, &cfg.size_factors)?;

    let (base_hist, total_records) = build_sparse_histogram(&chunk_paths, &qis, &initial_ri)?;
    let ola2 = find_ola2_best_rf_detailed(
        &qis,
        &base_hist,
        &initial_ri,
        &cfg.size_factors,
        cfg.k,
        cfg.suppression_limit,
        total_records,
    )?;
    let final_rf = ola2.best_rf.clone();
    let lowest_dm_star = ola2.lowest_dm_star;
    let num_equivalence_classes = ola2.num_equivalence_classes;

    let output_dir_path = root.join(&cfg.output_directory);
    generalize_and_write_outputs(
        &chunk_paths,
        &qis,
        &final_rf,
        cfg.k,
        &output_dir_path,
        &cfg.output_path,
    )?;

    ensure_output_dir(&output_dir_path)?;
    let eq_stats = equivalence_class_stats(&base_hist, &final_rf, cfg.k);
    fs::write(
        output_dir_path.join("equivalence_class_stats.json"),
        serde_json::to_string_pretty(&eq_stats)?,
    )?;
    fs::write(
        output_dir_path.join("top_ola2_nodes.json"),
        serde_json::to_string_pretty(&ola2.top_nodes)?,
    )?;

    Ok(StatusPayload {
        status: "success".to_string(),
        outputs: Some(json!({
            "phase": "rust_pipeline",
            "source_config": cfg.source_json_config.display().to_string(),
            "enable_k_anonymity": cfg.enable_k_anonymity,
            "output_path": cfg.output_path,
            "detected_csv_count": csvs.len(),
            "rows_per_chunk": rows_per_chunk,
            "chunk_count": chunk_paths.len(),
            "numerical_qi_count": cfg.numerical_qis.len(),
            "encoding_map_count": 0,
            "dynamic_min_max_count": dynamic_min_max.len(),
            "quasi_identifier_count": qis.len(),
            "max_equivalence_classes": max_eq,
            "initial_ri": initial_ri,
            "final_rf": final_rf,
            "lowest_dm_star": lowest_dm_star,
            "num_equivalence_classes": num_equivalence_classes,
            "equivalence_class_stats": eq_stats,
            "top_ola2_nodes": ola2.top_nodes,
            "total_records": total_records,
            "final_output_path": if Path::new(&cfg.output_path).is_absolute() {
                cfg.output_path.clone()
            } else {
                output_dir_path.join(&cfg.output_path).display().to_string()
            }
        })),
        error: None,
        log_file: "log.txt".to_string(),
    })
}

fn estimate_available_ram_bytes() -> Option<i64> {
    let raw = fs::read_to_string("/proc/meminfo").ok()?;
    for line in raw.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb = rest
                .split_whitespace()
                .next()
                .and_then(|s| s.parse::<i64>().ok())?;
            return Some(kb.saturating_mul(1024));
        }
    }
    None
}
