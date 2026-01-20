use std::collections::HashMap;
use ndarray::{ArrayD, Axis, IxDyn};

#[derive(Debug)]
pub struct OLA2 {
    pub tree: Vec<Vec<Vec<usize>>>,
    pub node_status: HashMap<Vec<usize>, bool>,

    pub lowest_dm_star: i64,
    pub best_rf: Option<Vec<usize>>,
    pub best_num_eq: usize,

    pub suppression_count: usize,
}

impl OLA2 {
    pub fn new() -> Self {
        Self {
            tree: Vec::new(),
            node_status: HashMap::new(),
            lowest_dm_star: i64::MAX,
            best_rf: None,
            best_num_eq: 0,
            suppression_count: 0,
        }
    }

    // -------------------------------------------------
    // Build RF lattice
    // -------------------------------------------------
    pub fn build_tree(
        &mut self,
        initial: &[usize],
        max_levels: &[usize],
        factor: usize,
    ) {
        self.tree.clear();
        self.tree.push(vec![initial.to_vec()]);
        self.node_status.insert(initial.to_vec(), false);

        loop {
            let mut next = Vec::new();
            for node in self.tree.last().unwrap() {
                for i in 0..node.len() {
                    let mut new_node = node.clone();
                    new_node[i] = usize::min(new_node[i] * factor, max_levels[i]);

                    if !self.node_status.contains_key(&new_node) {
                        self.node_status.insert(new_node.clone(), false);
                        next.push(new_node);
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            self.tree.push(next);
        }
    }

    // -------------------------------------------------
    // Merge histogram for a node
    // -------------------------------------------------
    pub fn merge_histogram(
        histogram: &ArrayD<i64>,
        node: &[usize],
    ) -> ArrayD<i64> {
        let mut out = histogram.clone();

        for (axis, &group) in node.iter().enumerate() {
            let mut shape = out.shape().to_vec();
            let pad = (group - shape[axis] % group) % group;

            if pad > 0 {
                shape[axis] += pad;
                out = out.broadcast(IxDyn(&shape)).unwrap().to_owned();
            }

            let mut new_shape = out.shape().to_vec();
            new_shape[axis] /= group;
            new_shape.insert(axis + 1, group);

            out = out
                .into_shape(IxDyn(&new_shape))
                .unwrap()
                .sum_axis(Axis(axis + 1));
        }
        out
    }

    // -------------------------------------------------
    // EXACT Python k-anon + suppression logic
    // -------------------------------------------------
    pub fn passes_k_with_suppression(
        &mut self,
        histogram: &ArrayD<i64>,
        k: usize,
        suppression_limit: usize,
    ) -> bool {
        self.suppression_count = 0;

        for &count in histogram.iter() {
            if count > 0 && (count as usize) < k {
                self.suppression_count += count as usize;
            }
        }

        self.suppression_count <= suppression_limit
    }

    // -------------------------------------------------
    // DM*
    // -------------------------------------------------
    pub fn dm_star(histogram: &ArrayD<i64>, k: usize) -> i64 {
        histogram
            .iter()
            .filter(|&&v| v as usize >= k)
            .map(|&v| v * v)
            .sum()
    }
}
