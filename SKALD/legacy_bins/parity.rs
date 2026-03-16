use std::fs;
use serde::Deserialize;

#[derive(Deserialize)]
struct PythonResult {
    final_rf: Vec<usize>,
    lowest_dm_star: i64,
    num_equivalence_classes: usize,
}

#[derive(Deserialize)]
struct RustResult {
    best_rf: Vec<usize>,
    lowest_dm_star: i64,
    num_equivalence_classes: usize,
}

fn main() {
    let py: PythonResult = serde_json::from_str(
        &fs::read_to_string("debug/python_result.json").unwrap(),
    ).unwrap();

    let rs: RustResult = serde_json::from_str(
        &fs::read_to_string("debug/rust_result.json").unwrap(),
    ).unwrap();

    println!("=== PARITY CHECK ===");
    println!("Python RF: {:?}", py.final_rf);
    println!("Rust RF  : {:?}", rs.best_rf);

    assert_eq!(py.final_rf, rs.best_rf, "RF mismatch");
    assert_eq!(py.lowest_dm_star, rs.lowest_dm_star, "DM* mismatch");
    assert_eq!(
        py.num_equivalence_classes,
        rs.num_equivalence_classes,
        "EQ class mismatch"
    );

    println!("✅ FULL OLA-2 PARITY ACHIEVED");
}
