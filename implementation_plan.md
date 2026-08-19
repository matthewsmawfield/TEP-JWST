# Consolidate Bayesian Evidence Tests

Update `step_176_nested_bayesian_evidence.py` and the manuscript to present a strong three-part result for the Bayesian Evidence, validating that the negative Occam penalty was due to biased mass absorption.

## Proposed Changes

### `scripts/steps/step_176_nested_bayesian_evidence.py`
#### [MODIFY] step_176_nested_bayesian_evidence.py
I will refactor the script to run THREE distinct regimes of tests, ensuring identical data, priors, and scaling across them:
1. **Conventional Comparison (Raw Mass):**
   - Use raw `mass` and `z` (standardized) for `Standard_Physics`.
   - Use `design_matrix_raw = [1, mass, z]` to residualize observables for the Residual Null.
   - This reproduces the original negative Bayes factor (e.g., ln BF ≈ -15.7) to demonstrate what happens when observed mass is assumed unbiased and allowed to absorb the $\Gamma_t$ signal.
2. **Incremental Test (Augmented Joint Test):**
   - Compares the `TEP_Augmented` model (which gets `mass`, `z`, and `log_gamma`) against the `Standard_Physics` model (which gets only `mass` and `z`).
   - This tests whether $\Gamma_t$ adds information beyond raw mass and redshift.
3. **TEP-Aware Comparison (Orthogonalized Mass):**
   - Use `mass_ortho` (mass orthogonalized against `log_gamma`) for `Corrected_Standard_Physics` or as the baseline for the residual tests.
   - Use `design_matrix_ortho = [1, mass_ortho, z_ortho]` for the residual tests.
   - This reproduces the ln BF ≈ +99.8 result, showing what happens when the predicted mass contamination is strictly removed.

### `site/components/4_results.html` & `site/components/5_discussion.html`
#### [MODIFY] 4_results.html
#### [MODIFY] 5_discussion.html
I will update the text describing the Bayesian evidence to formally structure it around these three tests. I will clearly explain:
- Why the **Conventional comparison** yields a penalty (collinearity with biased $M_{*,\rm obs}$ absorbs the signal).
- Why the **Incremental test** shows $\Gamma_t$ still adds value.
- Why the **TEP-aware comparison** flips the Bayes factor to +99.8 by preventing this illegal absorption.

## Verification Plan
1. Run `python3 scripts/steps/step_176_nested_bayesian_evidence.py` to ensure all 3 regimes execute successfully and produce the expected Bayes Factors (-15 and +99).
2. Run `npm run build:markdown` and `python3 scripts/steps/step_160_manuscript_consistency_check.py` to verify the manuscript text matches the pipeline outputs.
