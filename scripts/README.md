# TEP-JWST Analysis Pipeline

Reproducible analysis pipeline for testing the Temporal Equivalence Principle (TEP) using JWST high-redshift galaxy observations.

## Quick Start

```bash
python scripts/steps/run_all_steps.py
```

This runs all 38 analysis steps and produces results in `results/outputs/`.

## The TEP Model

All predictions derive from the exponential TEP equation:

```
Γ_t = exp[α(z) × (2/3) × (log M_h - 12) × z_factor]
```

where:
- `α(z) = α₀ × √(1+z)` is the redshift-dependent coupling
- `α₀ = 0.58 ± 0.16` (from Cepheid calibration, Paper 12)
- `log M_h` is the log halo mass (M_h = M* × 100)
- `z_factor = (1+z) / (1+z_ref)` with z_ref = 5.5

## Pipeline Structure

```
scripts/
├── steps/           # 38 analysis steps (run in order)
│   ├── run_all_steps.py    # Master runner
│   ├── step_01_*.py        # Data loading
│   ├── step_02_*.py        # TEP model
│   └── ...
└── utils/           # Shared utilities
    ├── tep_model.py        # TEP functions (NEW)
    ├── logger.py           # Logging
    └── style.py            # Plotting
```

## Pipeline Phases

### Phase I: Core Pipeline (Steps 00-07)
| Step | Script | Purpose |
|------|--------|---------|
| 01 | `step_01_uncover_load.py` | Load UNCOVER DR4, apply quality cuts |
| 02 | `step_02_tep_model.py` | Apply TEP model, compute Γ_t |
| 00 | `step_00_first_principles.py` | First-principles derivation |
| 03 | `step_03_thread1_z7_inversion.py` | Thread 1: z>7 mass-sSFR inversion |
| 04 | `step_04_thread2_4_partial_correlations.py` | Threads 2-4: Partial correlations |
| 05 | `step_05_thread5_z8_dust.py` | Thread 5: z>8 dust anomaly |
| 06 | `step_06_thread6_7_coherence.py` | Threads 6-7: Coherence tests |
| 07 | `step_07_summary.py` | Seven threads summary |

### Phase II: Cross-Domain Validation (Steps 08-12)
| Step | Script | Purpose |
|------|--------|---------|
| 08 | `step_08_holographic_synthesis.py` | Cross-paper consistency |
| 09 | `step_09_sn_ia_mass_step.py` | SN Ia mass step prediction |
| 10 | `step_10_mw_gc_gradient.py` | MW GC screening test |
| 11 | `step_11_sn_ia_extended.py` | Extended SN Ia analysis |
| 12 | `step_12_trgb_cepheid.py` | TRGB-Cepheid offset |
| 82 | `step_82_final_literature_tests.py` | Final literature tests |
| 83 | `step_83_cutting_edge_tests.py` | Cutting edge tests |
| 84 | `step_84_emission_line_tests.py` | Emission line tests |
| 85 | `step_85_final_comprehensive.py` | FINAL COMPREHENSIVE SUMMARY |
| 86 | `step_86_observational_signatures.py` | Observational signatures |
| 87 | `step_87_compactness_verification.py` | Compactness verification (LRDs) |
| 88 | `step_88_binary_pulsar_constraints.py` | Binary pulsar screening |
| 89 | `step_89_bbn_analysis.py` | BBN compatibility analysis |

### Phase III: JWST Extended Analysis (Steps 13-18)
| Step | Script | Purpose |
|------|--------|---------|
| 13 | `step_13_jwst_uv_slope.py` | UV slope analysis |
| 14 | `step_14_jwst_impossible_galaxies.py` | "Impossible" galaxies resolution |
| 15 | `step_15_robustness_tests.py` | Robustness and systematics |
| 16 | `step_16_ml_ratio.py` | Mass-to-light ratio (ρ = 0.97) |
| 17 | `step_17_assembly_time.py` | Assembly time (ρ = 1.00) |
| 18 | `step_18_chi2_analysis.py` | χ² correlation analysis |

### Phase IV: Statistical Synthesis (Steps 19-24)
| Step | Script | Purpose |
|------|--------|---------|
| 19 | `step_19_combined_significance.py` | Combined significance |
| 20 | `step_20_parameter_validation.py` | α₀ = 0.58 validation |
| 21 | `step_21_scatter_reduction.py` | Scatter reduction |
| 22 | `step_22_extreme_population.py` | Extreme population analysis |
| 23 | `step_23_self_consistency.py` | Self-consistency tests |
| 24 | `step_24_cosmological_implications.py` | Cosmological implications |

### Phase V: Final Validation (Steps 25-30)
| Step | Script | Purpose |
|------|--------|---------|
| 25 | `step_25_cross_sample_validation.py` | Cross-sample validation |
| 26 | `step_26_prediction_alignment.py` | Prediction-observation alignment |
| 27 | `step_27_multi_angle_validation.py` | Multi-angle validation |
| 28 | `step_28_chi2_diagnostic.py` | χ² as TEP diagnostic |
| 29 | `step_29_final_synthesis.py` | Final synthesis |
| 30 | `step_30_model_comparison.py` | AIC/BIC model comparison |

### Phase VI: Independent Replication (Steps 31-37)
| Step | Script | Purpose |
|------|--------|---------|
| 31 | `step_31_independent_validation.py` | Out-of-sample validation |
| 32 | `step_32_z8_dust_prediction.py` | z>8 dust quantitative tests |
| 33 | `step_33_ceers_replication.py` | CEERS replication (ρ = +0.68) |
| 34 | `step_34_ceers_download.py` | CEERS catalog download |
| 35 | `step_35_cosmosweb_download.py` | COSMOS-Web catalog download |
| 36 | `step_36_cosmosweb_replication.py` | COSMOS-Web replication (ρ = +0.63) |
| 37 | `step_37_spectroscopic_validation.py` | Spectroscopic validation (N=147) |

### Phase VII: Refinement and Robustness (Steps 37c-40)
| Step | Script | Purpose |
|------|--------|---------|
| 37c | `step_37c_spectroscopic_refinement.py` | Simpson's paradox tests |
| 38 | `step_38_resolved_gradients.py` | Resolved core screening |
| 39 | `step_39_environment_screening.py` | Environmental screening |
| 40 | `step_40_sensitivity_analysis.py` | Parameter sensitivity sweep |

### Phase VIII: Simulations (Visualizations)
| Script | Purpose |
|--------|---------|
| `simulations/predict_balmer_lines.py` | Spectroscopic prediction simulation |
| `simulations/plot_sensitivity.py` | Sensitivity analysis plot |
| `simulations/plot_screening_schematic.py` | Screening mechanism schematic |

## Key Results

### The Seven Threads (All Significant)

| Thread | Test | Result | p-value |
|--------|------|--------|---------|
| 1 | z>7 Mass-sSFR Inversion | Δρ = +0.25 | < 10⁻⁵ |
| 2 | Γ_t vs Age Ratio | ρ = +0.19 | < 10⁻²¹ |
| 3 | Γ_t vs Metallicity | ρ = +0.19 | < 10⁻¹¹ |
| 4 | Γ_t vs Dust | ρ = +0.54 | < 10⁻⁵⁰ |
| 5 | z>8 Dust Anomaly | ρ = +0.56 | < 10⁻²⁴ |
| 6 | Age-Metallicity Coherence | ρ = +0.33 | < 10⁻²⁹ |
| 7 | Multi-Property Split | All p < 10⁻¹⁰ | — |

### Partial Correlation Results

| Correlation | ρ | Status |
|-------------|---|--------|
| Γ_t vs Age Ratio | ~0.00 | Vanishes |
| Γ_t vs Metallicity | ~0.00 | Vanishes |
| z>8 Dust (Mass Control) | **+0.28** | **Robust** |

### Three-Survey Replication

| Survey | ρ(M*, dust) | N |
|--------|-------------|---|
| UNCOVER | +0.56 | 283 |
| CEERS | +0.68 | 82 |
| COSMOS-Web | +0.63 | 918 |
| **Total** | — | **1,283** |

## Outputs

Results saved to `results/outputs/`:
- `step_07_seven_threads_summary.json` — Main summary
- `step_02_tep_model_summary.json` — TEP model parameters
- `step_33_ceers_replication.json` — CEERS results
- `step_36_cosmosweb_replication.json` — COSMOS-Web results
- `step_37_spectroscopic_validation.json` — Spectroscopic validation

Intermediate data saved to `results/interim/`:
- `step_01_uncover_full_sample.csv` — Raw sample
- `step_02_uncover_full_sample_tep.csv` — With Γ_t computed

## Methodology Notes

### Partial Correlation
- **z-only control**: Linear residualization of both variables against z
- **Double-control**: Uses `log(Γ_t)` for proper residualization of exponential form

### Robustness Tests
- Permutation test: p < 0.0001
- Bootstrap 95% CI: [0.46, 0.64]
- Cohen's d = 0.89 (large effect)
- Fisher's exact: OR = 5.25
- Outlier robust: ρ = 0.45 with 10% trimming

## Caveats

1. **z = 6–7 anomaly**: Negative mass-dust correlation (ρ = −0.12, p = 0.02) at this redshift
2. **Spectroscopic validation**: Marginal (ρ = +0.18, p = 0.047)
3. **Mass circularity**: Γ_t is derived from M*, limiting independent tests
