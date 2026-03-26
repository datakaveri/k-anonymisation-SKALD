use crate::pipeline::bootstrap::{
    http_status_for, suggested_fix_for, ErrorPayload, PipelineError, StatusPayload,
};
use crate::pipeline::pipeline::run_pipeline;
use std::fs;
use std::path::PathBuf;

pub fn main_entry() {
    let root = PathBuf::from(".");
    let status = match run_pipeline(&root) {
        Ok(ok) => ok,

        Err(PipelineError::Validation { code, message, details }) => {
            let code_str = code.to_string();
            StatusPayload {
                status: "error".to_string(),
                phase: None,
                outputs: None,
                error: Some(ErrorPayload {
                    suggested_fix: suggested_fix_for(code).to_string(),
                    http_status_code: http_status_for(code),
                    code: code_str,
                    message,
                    details,
                }),
                log_file: "output/pipeline.log".to_string(),
            }
        }

        Err(PipelineError::Io(e)) => {
            let (code, msg) = match e.kind() {
                std::io::ErrorKind::NotFound => (
                    "IO_READ_FAILED",
                    "A required file or directory was not found",
                ),
                std::io::ErrorKind::PermissionDenied => (
                    "IO_PERMISSION_DENIED",
                    "Permission denied accessing a file or directory",
                ),
                std::io::ErrorKind::WriteZero
                | std::io::ErrorKind::StorageFull => (
                    "IO_WRITE_FAILED",
                    "Failed to write output — disk may be full",
                ),
                _ => ("IO_READ_FAILED", "An unexpected I/O error occurred"),
            };
            StatusPayload {
                status: "error".to_string(),
                phase: None,
                outputs: None,
                error: Some(ErrorPayload {
                    suggested_fix: suggested_fix_for(code).to_string(),
                    http_status_code: http_status_for(code),
                    code: code.to_string(),
                    message: msg.to_string(),
                    details: e.to_string(),
                }),
                log_file: "output/pipeline.log".to_string(),
            }
        }

        Err(PipelineError::Json(e)) => {
            let code = "CONFIG_PARSE_ERROR";
            StatusPayload {
                status: "error".to_string(),
                phase: None,
                outputs: None,
                error: Some(ErrorPayload {
                    suggested_fix: suggested_fix_for(code).to_string(),
                    http_status_code: http_status_for(code),
                    code: code.to_string(),
                    message: "Failed to parse JSON configuration".to_string(),
                    details: e.to_string(),
                }),
                log_file: "output/pipeline.log".to_string(),
            }
        }
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
