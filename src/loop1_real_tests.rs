//! Explicit real-data validation for the streamed one-loop fixed-policy kernels.
//!
//! All four ZXCL_VALIDATION_* inputs must be supplied explicitly. Source graph,
//! membership, policy, and value fixtures remain read-only; the probe owns any
//! temporary fixtures. It compares every L9 state and every state in coherent
//! whole L8 blocks using both reward settings, actual L0 values, and independent
//! signed vectors.

use super::*;

fn required_path(name: &str) -> Result<PathBuf> {
    env::var_os(name)
        .map(PathBuf::from)
        .with_context(|| format!("explicit streamed one-loop validation requires {name}"))
}

#[test]
#[ignore = "requires explicit ZXCL_VALIDATION_GRAPH, PRUNED, POLICY, and VALUES fixtures"]
fn streamed_l9_and_l8_match_production_fixed_policy() -> Result<()> {
    let graph = required_path("ZXCL_VALIDATION_GRAPH")?;
    let pruned = required_path("ZXCL_VALIDATION_PRUNED")?;
    let policy = required_path("ZXCL_VALIDATION_POLICY")?;
    let values = required_path("ZXCL_VALIDATION_VALUES")?;
    let l0file = if values.is_dir() {
        mdp_f32_value_path(&values, 0)
    } else {
        values
    };
    let max_states = match env::var("ZXCL_LOOP1_MAX_STATES") {
        Ok(value) => value
            .parse::<usize>()
            .context("ZXCL_LOOP1_MAX_STATES must be an integer in 1..=4000000")?,
        Err(env::VarError::NotPresent) => 2_000_000,
        Err(error) => return Err(error).context("read ZXCL_LOOP1_MAX_STATES"),
    };
    anyhow::ensure!(
        (1..=4_000_000).contains(&max_states),
        "ZXCL_LOOP1_MAX_STATES must be in 1..=4000000"
    );

    crate::loop1_probe::validate_real(&graph, &pruned, &policy, &l0file, max_states)
}
