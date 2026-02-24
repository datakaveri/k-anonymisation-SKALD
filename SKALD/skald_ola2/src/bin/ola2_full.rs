use std::collections::HashSet;
use std::env;
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use ndarray::{ArrayD, IxDyn};
use serde::{Deserialize, Serialize};

#[derive(Debug)]
enum Ola2Error {
    Io(std::io::Error),
    Json(serde_json::Error),
    InvalidInput(String),
}

impl fmt::Display for Ola2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Ola2Error::Io(e) => write!(f, "I/O error: {e}"),
            Ola2Error::Json(e) => write!(f, "JSON error: {e}"),
            Ola2Error::InvalidInput(msg) => write!(f, "Invalid input: {msg}"),
        }
    }
}

impl Error for Ola2Error {}

impl From<std::io::Error> for Ola2Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for Ola2Error {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

#[derive(Debug, Deserialize)]
struct PythonInput {
    initial_ri: Vec<usize>,
    max_levels: Vec<usize>,
    k: usize,
    suppression_limit: f64,
    total_records: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct RustResult {
    best_rf: Option<Vec<usize>>,
    lowest_dm_star: Option<i64>,
    num_equivalence_classes: Option<usize>,
    elapsed_time: f64,
}

#[derive(Debug)]
struct CliPaths {
    input_path: PathBuf,
    shape_path: PathBuf,
    flat_path: PathBuf,
    output_path: PathBuf,
}

fn parse_paths(args: &[String]) -> Result<CliPaths, Ola2Error> {
    if args.is_empty() {
        return Ok(CliPaths {
            input_path: PathBuf::from("debug/python_result.json"),
            shape_path: PathBuf::from("debug/global_hist_shape.json"),
            flat_path: PathBuf::from("debug/global_hist_flat.json"),
            output_path: PathBuf::from("debug/rust_result.json"),
        });
    }

    if args.len() != 4 {
        return Err(Ola2Error::InvalidInput(
            "Usage: ola2_full [input_json shape_json flat_json output_json]".to_string(),
        ));
    }

    Ok(CliPaths {
        input_path: PathBuf::from(&args[0]),
        shape_path: PathBuf::from(&args[1]),
        flat_path: PathBuf::from(&args[2]),
        output_path: PathBuf::from(&args[3]),
    })
}

fn validate_python_input(input: &PythonInput) -> Result<(), Ola2Error> {
    if input.initial_ri.is_empty() {
        return Err(Ola2Error::InvalidInput(
            "initial_ri cannot be empty".to_string(),
        ));
    }
    if input.initial_ri.len() != input.max_levels.len() {
        return Err(Ola2Error::InvalidInput(
            "initial_ri and max_levels length mismatch".to_string(),
        ));
    }
    if input.initial_ri.iter().any(|&v| v == 0) {
        return Err(Ola2Error::InvalidInput(
            "initial_ri must contain only positive values".to_string(),
        ));
    }
    if input.k == 0 {
        return Err(Ola2Error::InvalidInput("k must be > 0".to_string()));
    }
    if input.total_records == 0 {
        return Err(Ola2Error::InvalidInput(
            "total_records must be > 0".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&input.suppression_limit) {
        return Err(Ola2Error::InvalidInput(
            "suppression_limit must be in [0, 1]".to_string(),
        ));
    }
    if input.max_levels.iter().any(|&v| v > (usize::BITS as usize - 2)) {
        return Err(Ola2Error::InvalidInput(
            "max_levels contains a value too large to shift".to_string(),
        ));
    }
    Ok(())
}

fn build_lattice_levels(
    initial: &[usize],
    max_levels_pow: &[usize],
) -> Result<Vec<Vec<Vec<usize>>>, Ola2Error> {
    let mut tree = vec![vec![initial.to_vec()]];
    let mut seen = HashSet::new();
    seen.insert(initial.to_vec());

    let mut max_levels = Vec::with_capacity(max_levels_pow.len());
    for &pow in max_levels_pow {
        let shifted = 1usize.checked_shl(pow as u32).ok_or_else(|| {
            Ola2Error::InvalidInput("max_levels shift overflow".to_string())
        })?;
        max_levels.push(shifted);
    }

    for (i, &start) in initial.iter().enumerate() {
        if start > max_levels[i] {
            return Err(Ola2Error::InvalidInput(format!(
                "initial_ri[{i}]={start} exceeds max level {}",
                max_levels[i]
            )));
        }
    }

    loop {
        let mut next = Vec::new();

        for node in tree
            .last()
            .ok_or_else(|| Ola2Error::InvalidInput("tree construction failed".to_string()))?
        {
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

    Ok(tree)
}

fn merge_histogram(hist: &ArrayD<i64>, rf: &[usize]) -> Result<ArrayD<i64>, Ola2Error> {
    let ndim = hist.ndim();
    if rf.len() != ndim {
        return Err(Ola2Error::InvalidInput(
            "rf length must match histogram dimensions".to_string(),
        ));
    }
    if rf.iter().any(|&v| v == 0) {
        return Err(Ola2Error::InvalidInput(
            "rf must contain only positive values".to_string(),
        ));
    }

    let mut out_shape = Vec::with_capacity(ndim);
    for (axis, &bw) in rf.iter().enumerate() {
        let size = hist.shape()[axis];
        let bins = size.div_ceil(bw);
        out_shape.push(bins);
    }

    let mut out = ArrayD::<i64>::zeros(IxDyn(&out_shape));
    let mut out_idx = vec![0usize; ndim];

    for (idx, v) in hist.indexed_iter() {
        for i in 0..ndim {
            out_idx[i] = idx[i] / rf[i];
        }

        let slot = out.get_mut(IxDyn(&out_idx)).ok_or_else(|| {
            Ola2Error::InvalidInput("failed writing merged histogram index".to_string())
        })?;
        *slot += *v;
    }

    Ok(out)
}

fn passes_k_or_suppression(
    hist: &ArrayD<i64>,
    k: usize,
    suppression_limit: f64,
    total_records: usize,
) -> bool {
    let mut suppressed = 0usize;
    for &v in hist {
        if v > 0 && (v as usize) < k {
            suppressed += v as usize;
        }
    }
    let allowed = (suppression_limit * total_records as f64).floor() as usize;
    suppressed <= allowed
}

fn compute_dm_star(hist: &ArrayD<i64>, k: usize) -> (i64, usize) {
    let mut dm = 0i64;
    let mut eq = 0usize;
    let mut suppressed = 0i64;

    for &v in hist {
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

fn load_json<T: for<'de> Deserialize<'de>>(path: &PathBuf) -> Result<T, Ola2Error> {
    let raw = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&raw)?)
}

fn load_histogram(shape_path: &PathBuf, flat_path: &PathBuf) -> Result<ArrayD<i64>, Ola2Error> {
    let shape: Vec<usize> = load_json(shape_path)?;
    let flat: Vec<i64> = load_json(flat_path)?;

    if shape.is_empty() {
        return Err(Ola2Error::InvalidInput(
            "histogram shape is empty".to_string(),
        ));
    }
    let expected: usize = shape.iter().product();
    if expected != flat.len() {
        return Err(Ola2Error::InvalidInput(format!(
            "histogram flat length mismatch: expected {expected}, got {}",
            flat.len()
        )));
    }

    ArrayD::from_shape_vec(IxDyn(&shape), flat)
        .map_err(|e| Ola2Error::InvalidInput(format!("invalid histogram shape: {e}")))
}

fn compute_best_rf(input: &PythonInput, base_hist: &ArrayD<i64>) -> Result<RustResult, Ola2Error> {
    validate_python_input(input)?;
    let tree = build_lattice_levels(&input.initial_ri, &input.max_levels)?;

    let start = Instant::now();
    let mut best_rf = None;
    let mut best_dm = i64::MAX;
    let mut best_eq = None;

    for rf in tree.iter().flatten() {
        let merged = merge_histogram(base_hist, rf)?;
        let pass = passes_k_or_suppression(&merged, input.k, input.suppression_limit, input.total_records);
        if pass {
            let (dm, eq) = compute_dm_star(&merged, input.k);
            if dm < best_dm {
                best_dm = dm;
                best_rf = Some(rf.clone());
                best_eq = Some(eq);
            }
        }
    }

    Ok(RustResult {
        best_rf,
        lowest_dm_star: if best_dm == i64::MAX { None } else { Some(best_dm) },
        num_equivalence_classes: best_eq,
        elapsed_time: start.elapsed().as_secs_f64(),
    })
}

fn run_with_paths(paths: &CliPaths) -> Result<RustResult, Ola2Error> {
    let input: PythonInput = load_json(&paths.input_path)?;
    let base_hist = load_histogram(&paths.shape_path, &paths.flat_path)?;
    let result = compute_best_rf(&input, &base_hist)?;

    let payload = serde_json::to_string_pretty(&result)?;
    fs::write(&paths.output_path, payload)?;
    Ok(result)
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let paths = match parse_paths(&args) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("ola2_full error: {e}");
            std::process::exit(1);
        }
    };

    match run_with_paths(&paths) {
        Ok(result) => {
            println!("=== RUST OLA-2 RESULT ===");
            println!("{:#?}", result);
        }
        Err(e) => {
            eprintln!("ola2_full error: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before epoch")
            .as_nanos();
        p.push(format!("skald_ola2_{nanos}_{name}"));
        p
    }

    #[test]
    fn merge_histogram_bins_correctly() {
        let hist = array![[1, 2, 3, 4], [5, 6, 7, 8]].into_dyn();
        let rf = vec![2, 2];
        let merged = merge_histogram(&hist, &rf).unwrap();
        let expected = array![[14, 22]].into_dyn();
        assert_eq!(merged, expected);
    }

    #[test]
    fn dm_star_matches_definition() {
        let hist = array![[2, 1], [0, 3]].into_dyn();
        let (dm, eq) = compute_dm_star(&hist, 2);
        assert_eq!(eq, 2);
        assert_eq!(dm, 14);
    }

    #[test]
    fn suppression_limit_check() {
        let hist = array![[1, 2], [0, 0]].into_dyn();
        assert!(passes_k_or_suppression(&hist, 2, 0.25, 4));
    }

    #[test]
    fn merge_histogram_rejects_mismatched_dims() {
        let hist = array![[1, 2], [3, 4]].into_dyn();
        let err = merge_histogram(&hist, &[2]).unwrap_err();
        assert!(matches!(err, Ola2Error::InvalidInput(_)));
    }

    #[test]
    fn parse_paths_defaults() {
        let paths = parse_paths(&[]).unwrap();
        assert_eq!(paths.input_path, PathBuf::from("debug/python_result.json"));
    }

    #[test]
    fn parse_paths_rejects_invalid_arity() {
        let err = parse_paths(&["only_one".to_string()]).unwrap_err();
        assert!(matches!(err, Ola2Error::InvalidInput(_)));
    }

    #[test]
    fn validate_python_input_rejects_bad_k() {
        let input = PythonInput {
            initial_ri: vec![1],
            max_levels: vec![5],
            k: 0,
            suppression_limit: 0.0,
            total_records: 1,
        };
        let err = validate_python_input(&input).unwrap_err();
        assert!(matches!(err, Ola2Error::InvalidInput(_)));
    }

    #[test]
    fn compute_best_rf_small_hist() {
        let input = PythonInput {
            initial_ri: vec![1],
            max_levels: vec![2],
            k: 2,
            suppression_limit: 0.0,
            total_records: 10,
        };
        let hist = array![1, 1, 4, 4].into_dyn();
        let result = compute_best_rf(&input, &hist).unwrap();
        assert_eq!(result.best_rf, Some(vec![2]));
        assert_eq!(result.lowest_dm_star, Some(68));
        assert_eq!(result.num_equivalence_classes, Some(2));
    }

    #[test]
    fn validate_python_input_rejects_bad_suppression_limit() {
        let input = PythonInput {
            initial_ri: vec![1],
            max_levels: vec![5],
            k: 2,
            suppression_limit: 1.5,
            total_records: 1,
        };
        let err = validate_python_input(&input).unwrap_err();
        assert!(matches!(err, Ola2Error::InvalidInput(_)));
    }

    #[test]
    fn load_histogram_rejects_shape_flat_mismatch() {
        let shape_path = temp_path("shape.json");
        let flat_path = temp_path("flat.json");

        fs::write(&shape_path, "[2,2]").expect("shape write failed");
        fs::write(&flat_path, "[1,2,3]").expect("flat write failed");

        let err = load_histogram(&shape_path, &flat_path).unwrap_err();
        assert!(matches!(err, Ola2Error::InvalidInput(_)));

        let _ = fs::remove_file(shape_path);
        let _ = fs::remove_file(flat_path);
    }

    #[test]
    fn compute_best_rf_rejects_initial_ri_gt_max() {
        let input = PythonInput {
            initial_ri: vec![8],
            max_levels: vec![2], // max = 4
            k: 2,
            suppression_limit: 0.0,
            total_records: 10,
        };
        let hist = array![1, 1, 4, 4].into_dyn();
        let err = compute_best_rf(&input, &hist).unwrap_err();
        assert!(matches!(err, Ola2Error::InvalidInput(_)));
    }
}
