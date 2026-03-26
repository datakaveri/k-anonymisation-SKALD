use crate::pipeline::anonymization::{
    build_quasi_identifiers, build_sparse_histogram, compute_numerical_min_max, equivalence_class_stats,
    find_ola1_initial_ri, find_ola2_best_rf_detailed, generalize_and_write_outputs,
};
use crate::pipeline::bootstrap::{
    ensure_output_dir, find_first_json_config, parse_runtime_config, split_csv_by_ram,
    Logger, PipelineError, StatusPayload,
};
use crate::pipeline::preprocess::preprocess_chunks;
use serde_json::json;
use std::fs;
use std::path::Path;

pub fn run_pipeline(root: &Path) -> Result<StatusPayload, PipelineError> {
    let output_dir_path = root.join("output");
    let mut log = Logger::new(&output_dir_path);

    log.info("startup", "SKALD pipeline starting");

    log.info("config", "Searching for JSON config in config/");
    let config_path = find_first_json_config(&root.join("config"))?;
    log.info("config", &format!("Loaded config: {}", config_path.display()));
    let cfg = parse_runtime_config(&config_path)?;
    log.info("config", &format!(
        "k={}, suppression_limit={:.3}, k_anonymity={}",
        cfg.k, cfg.suppression_limit, cfg.enable_k_anonymity
    ));

    log.info("chunking", "Splitting CSV into RAM-sized chunks");
    let (chunk_paths, rows_per_chunk) = split_csv_by_ram(&root.join("data"), &root.join("chunks"), None, None)?;
    log.info("chunking", &format!(
        "{} chunk(s), ~{} rows/chunk",
        chunk_paths.len(),
        rows_per_chunk
    ));

    log.info("preprocessing", &format!(
        "Running preprocessing on {} chunk(s): suppress={}, hash_salt={}, hash={}, mask={}, encrypt={}, charcloak={}, tokenize={}, fpe={}",
        chunk_paths.len(),
        cfg.suppress.len(),
        cfg.hashing_with_salt.len(),
        cfg.hashing_without_salt.len(),
        cfg.masking.len(),
        cfg.encrypt.len(),
        cfg.charcloak.len(),
        cfg.tokenization.len(),
        cfg.fpe.len(),
    ));
    preprocess_chunks(&chunk_paths, &cfg)?;
    log.info("preprocessing", "Preprocessing complete");

    log.info("min_max_scan", &format!(
        "Computing min/max for {} numerical QI(s)",
        cfg.numerical_qis.len()
    ));
    let numerical_cols = cfg.numerical_qis.iter().map(|q| q.column.clone()).collect::<Vec<_>>();
    let dynamic_min_max = compute_numerical_min_max(&chunk_paths, &numerical_cols)?;
    log.info("min_max_scan", &format!("Scanned {} column(s)", dynamic_min_max.len()));

    log.info("qi_building", &format!(
        "Building QIs: {} numerical, {} categorical",
        cfg.numerical_qis.len(),
        cfg.categorical_qis.len()
    ));
    let qis = build_quasi_identifiers(&cfg, &dynamic_min_max)?;
    log.info("qi_building", &format!("{} quasi-identifier(s) built", qis.len()));

    let max_eq = estimate_available_ram_bytes().unwrap_or(32_000_000) / 32;
    log.info("ola1", &format!(
        "OLA-1: finding initial RI (max_eq={}, chunks={})",
        max_eq,
        chunk_paths.len()
    ));
    let initial_ri = find_ola1_initial_ri(&qis, chunk_paths.len() as i64, max_eq, &cfg.size_factors)?;
    log.info("ola1", &format!("Initial RI: {:?}", initial_ri));

    log.info("histogram", &format!(
        "Building sparse histogram over {} chunk(s)",
        chunk_paths.len()
    ));
    let (base_hist, total_records) = build_sparse_histogram(&chunk_paths, &qis, &initial_ri)?;
    log.info("histogram", &format!(
        "{} total records, {} histogram buckets",
        total_records,
        base_hist.len()
    ));

    log.info("ola2", &format!(
        "OLA-2: lattice search (k={}, suppression_limit={:.3})",
        cfg.k,
        cfg.suppression_limit
    ));
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
    log.info("ola2", &format!(
        "Best RF: {:?}, DM*={:.4}, ECs={}",
        final_rf, lowest_dm_star, num_equivalence_classes
    ));

    log.info("generalization", &format!(
        "Generalizing and writing output to {}",
        output_dir_path.display()
    ));
    generalize_and_write_outputs(
        &chunk_paths,
        &qis,
        &final_rf,
        cfg.k,
        &output_dir_path,
        &cfg.output_path,
    )?;
    log.info("generalization", "Output written");

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

    log.info("output", "Equivalence class stats and top OLA-2 nodes written");
    log.info("startup", "Pipeline completed successfully");

    let final_output_path = if Path::new(&cfg.output_path).is_absolute() {
        cfg.output_path.clone()
    } else {
        output_dir_path.join(&cfg.output_path).display().to_string()
    };

    Ok(StatusPayload {
        status: "success".to_string(),
        phase: Some("done".to_string()),
        outputs: Some(json!({
            "source_config": cfg.source_json_config.display().to_string(),
            "enable_k_anonymity": cfg.enable_k_anonymity,
            "output_path": cfg.output_path,
            "detected_csv_count": 1,
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
            "final_output_path": final_output_path,
        })),
        error: None,
        log_file: "output/pipeline.log".to_string(),
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
