use crate::pipeline::bootstrap::{ErrorPayload, PipelineError, StatusPayload};
use crate::pipeline::pipeline::run_pipeline;
use std::fs;
use std::path::PathBuf;

pub fn main_entry() {
    let root = PathBuf::from(".");
    let status = match run_pipeline(&root) {
        Ok(ok) => ok,
        Err(PipelineError::Validation { code, message, details }) => StatusPayload {
            status: "error".to_string(),
            outputs: None,
            error: Some(ErrorPayload {
                code: code.to_string(),
                message,
                details,
                suggested_fix: "Fix config/data and retry".to_string(),
            }),
            log_file: "log.txt".to_string(),
        },
        Err(e) => StatusPayload {
            status: "error".to_string(),
            outputs: None,
            error: Some(ErrorPayload {
                code: "INTERNAL_ERROR".to_string(),
                message: "Rust pipeline failed".to_string(),
                details: e.to_string(),
                suggested_fix: "Check logs and retry".to_string(),
            }),
            log_file: "log.txt".to_string(),
        },
    };

    let out_dir = PathBuf::from("output");
    if let Err(e) = fs::create_dir_all(&out_dir) {
        eprintln!("failed creating output directory: {e}");
        std::process::exit(1);
    }
    let status_path = out_dir.join("status.json");
    match serde_json::to_string_pretty(&status) {
        Ok(body) => {
            if let Err(e) = fs::write(&status_path, body) {
                eprintln!("failed writing output/status.json: {e}");
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("failed serializing status payload: {e}");
            std::process::exit(1);
        }
    }

    if status.status == "error" {
        std::process::exit(1);
    }
}
