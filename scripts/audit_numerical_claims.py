#!/usr/bin/env python3
"""Full audit of manuscript numerical claims against pipeline outputs."""
import json, os, re, sys

OUTPUTS_DIR = "results/outputs"

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return None

def search_all_jsons(pattern, key_filter=None):
    """Search all JSON files for a numeric pattern near a keyword."""
    results = []
    for fname in sorted(os.listdir(OUTPUTS_DIR)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(OUTPUTS_DIR, fname)
        try:
            with open(path) as f:
                s = f.read()
            for m in re.finditer(pattern, s):
                ctx = s[max(0, m.start()-60):m.end()+60]
                if key_filter is None or key_filter.lower() in ctx.lower():
                    results.append((fname, ctx.replace("\n", " ")))
        except:
            pass
    return results

# ============================================================
# 1. Verify ρ = +0.60 three-survey meta-analysis
# ============================================================
print("=" * 60)
print("1. THREE-SURVEY META-ANALYSIS ρ = +0.60")
print("=" * 60)
d = load_json(f"{OUTPUTS_DIR}/step_081_survey_cross_correlation.json")
if d:
    ma = d.get("meta_analysis", {})
    print(f"  rho_combined = {ma.get('rho_combined', 'N/A'):.4f} (manuscript: +0.60)")
    print(f"  z_stat = {ma.get('z_stat', 'N/A'):.2f} (manuscript: 24.6σ)")
    print(f"  p_combined = {ma.get('p_combined', 'N/A'):.4e}")
    print(f"  n_total = {ma.get('n_total', 'N/A')}")
    het = d.get("heterogeneity", {})
    print(f"  Q = {het.get('Q', 'N/A'):.2f} (manuscript: 11.2)")
    print(f"  I² = {het.get('I2', 'N/A'):.2f} (manuscript: 82%)")
    sc = d.get("survey_correlations", {})
    for survey, vals in sc.items():
        print(f"  {survey}: rho={vals.get('rho', 0):.4f}, p={vals.get('p', 0):.2e}, n={vals.get('n', 0)}")

# ============================================================
# 2. Verify SUSPENSE kinematic values
# ============================================================
print()
print("=" * 60)
print("2. SUSPENSE KINEMATIC (step_170)")
print("=" * 60)
d = load_json(f"{OUTPUTS_DIR}/step_170_kinematic_decisive_test.json")
if d:
    print(f"  N = {d.get('n_galaxies', 'N/A')}")
    def fmt(v, spec=".4f"):
        if isinstance(v, (int, float)):
            return f"{v:{spec}}"
        return str(v)
    print(f"  ρ(Age, Γ_t | z) = {fmt(d.get('partial_rho_gamma_dyn_age_given_z'))} (manuscript: +0.717)")
    print(f"  p = {fmt(d.get('p_partial_gamma_dyn_age_given_z'), '.4e')} (manuscript: 2.62×10⁻³)")
    print(f"  ρ(Age, M* | z) = {fmt(d.get('partial_rho_mstar_age_given_z'))} (manuscript: +0.493)")
    print(f"  p = {fmt(d.get('p_partial_mstar_age_given_z'), '.4e')} (manuscript: 0.062)")
    print(f"  ρ(Age, Γ_t | M*, z) = {fmt(d.get('partial_rho_gamma_dyn_age_given_mstar_z'))} (manuscript: +0.599)")
    print(f"  p = {fmt(d.get('p_partial_gamma_dyn_age_given_mstar_z'), '.4e')} (manuscript: 1.83×10⁻²)")
    print(f"  ρ(Age, M* | Γ_t, z) = {fmt(d.get('partial_rho_mstar_age_given_gamma_dyn_z'))} (manuscript: +0.025)")
    print(f"  p = {fmt(d.get('p_partial_mstar_age_given_gamma_dyn_z'), '.4e')}")

# ============================================================
# 3. Verify ρ(t_eff, A_V | t_cosmic) = +0.430
# ============================================================
print()
print("=" * 60)
print("3. PARTIAL ρ(t_eff, A_V | t_cosmic) = +0.430")
print("=" * 60)
results = search_all_jsons(r"0\.4[23]\d*", "partial")
for fname, ctx in results[:10]:
    print(f"  {fname}: {ctx}")

# Also check step_030
d = load_json(f"{OUTPUTS_DIR}/step_030_z8_dust_prediction.json")
if d:
    print(f"\n  step_030 keys: {list(d.keys())}")
    for k in d:
        if isinstance(d[k], dict):
            for k2 in d[k]:
                if "partial" in k2.lower() or "rho" in k2.lower():
                    print(f"    {k}.{k2} = {d[k][k2]}")

# ============================================================
# 4. Verify Fisher combination and meta-analysis
# ============================================================
print()
print("=" * 60)
print("4. FISHER COMBINATION (step_161)")
print("=" * 60)
d = load_json(f"{OUTPUTS_DIR}/step_161_multi_dataset_l1_combination.json")
if d:
    s = json.dumps(d)
    # Find z values
    for m in re.finditer(r'"(z_stat|z_combined|fisher_z|z_score)"[^,]*([\d.]+)', s):
        print(f"  {m.group(1)} = {m.group(2)}")
    for m in re.finditer(r'"(p_combined|p_value|fisher_p)"[^,]*([\d.e-]+)', s):
        print(f"  {m.group(1)} = {m.group(2)}")
    for m in re.finditer(r'"(chi2|chi_squared)"[^,]*([\d.]+)', s):
        print(f"  {m.group(1)} = {m.group(2)}")

# ============================================================
# 5. Verify Bayesian evidence (step_176)
# ============================================================
print()
print("=" * 60)
print("5. BAYESIAN EVIDENCE (step_176)")
print("=" * 60)
d = load_json(f"{OUTPUTS_DIR}/step_176_nested_bayesian_evidence.json")
if d:
    print(f"  keys: {list(d.keys())}")
    s = json.dumps(d, indent=2)
    # Print first 2000 chars
    print(s[:2000])
else:
    print("  File not found or empty")

# ============================================================
# 6. Verify Bonferroni floor (step_118)
# ============================================================
print()
print("=" * 60)
print("6. BONFERRONI FLOOR (step_118)")
print("=" * 60)
d = load_json(f"{OUTPUTS_DIR}/step_118_neff_corrected_significance.json")
if d:
    s = json.dumps(d)
    for m in re.finditer(r'"(bonferroni|sigma_floor|n_eff|corrected|floor)"[^,]*([\d.e-]+)', s, re.IGNORECASE):
        print(f"  {m.group(1)} = {m.group(2)}")
else:
    print("  File not found")

# ============================================================
# 7. Verify SMF resolution (step_146, step_174)
# ============================================================
print()
print("=" * 60)
print("7. SMF RESOLUTION (step_146, step_174)")
print("=" * 60)
d = load_json(f"{OUTPUTS_DIR}/step_146_stellar_mass_function_resolution.json")
if d:
    s = json.dumps(d)
    # Find percentage or fraction values
    for m in re.finditer(r'"(resolution|fraction|percent|excess|reduction|resolved)"[^,]*([\d.]+)', s, re.IGNORECASE):
        print(f"  step_146: {m.group(1)} = {m.group(2)}")
    print(f"  step_146 full (first 1000): {s[:1000]}")

d = load_json(f"{OUTPUTS_DIR}/step_174_smf_mass_threshold_counts.json")
if d:
    s = json.dumps(d)
    print(f"  step_174 (first 1000): {s[:1000]}")

# ============================================================
# 8. Verify SFE / Red Monster (step_043)
# ============================================================
print()
print("=" * 60)
print("8. SFE / RED MONSTER (step_043)")
print("=" * 60)
# Find step_043
for fname in sorted(os.listdir(OUTPUTS_DIR)):
    if "043" in fname:
        d = load_json(f"{OUTPUTS_DIR}/{fname}")
        if d:
            print(f"  {fname}: {json.dumps(d, indent=2)[:1000]}")

# ============================================================
# 9. Verify odds ratio and ΔAIC (step_136, step_141)
# ============================================================
print()
print("=" * 60)
print("9. ODDS RATIO & ΔAIC")
print("=" * 60)
d = load_json(f"{OUTPUTS_DIR}/step_136_functional_form_test.json")
if d:
    hl = d.get("headline", {})
    print(f"  step_136 rho_dust_teff_z8 = {hl.get('rho_dust_teff_z8', 'N/A'):.4f} (manuscript: +0.57)")
    print(f"  step_136 rho_dust_mstar_z8 = {hl.get('rho_dust_mstar_z8', 'N/A'):.4f} (manuscript: +0.53)")
    # Check for odds ratio
    s = json.dumps(d)
    for m in re.finditer(r'"(odds_ratio|odds|delta_aic|aic)"[^,]*([\d.]+)', s, re.IGNORECASE):
        print(f"  step_136: {m.group(1)} = {m.group(2)}")

d = load_json(f"{OUTPUTS_DIR}/step_141_nonlinear_aic.json")
if d:
    s = json.dumps(d)
    for m in re.finditer(r'"(delta_aic|odds_ratio|odds)"[^,]*([\d.]+)', s, re.IGNORECASE):
        print(f"  step_141: {m.group(1)} = {m.group(2)}")

# ============================================================
# 10. Verify σ₈ (step_135)
# ============================================================
print()
print("=" * 60)
print("10. σ₈ (step_135)")
print("=" * 60)
d = load_json(f"{OUTPUTS_DIR}/step_135_scale_dependent_growth.json")
if d:
    bf = d.get("best_fit", {})
    print(f"  sigma8_tep = {bf.get('sigma8_tep', 'N/A')}")
    print(f"  sigma8_planck = {bf.get('sigma8_planck', 'N/A')}")
    print(f"  delta_sigma8 = {bf.get('delta_sigma8', 'N/A')}")
    print(f"  delta_sigma8_in_sigma = {bf.get('delta_sigma8_in_sigma', 'N/A')}")
    ec = d.get("effective_coupling", {})
    print(f"  effective_coupling = {ec}")

# ============================================================
# 11. Verify dynamical mass comparison (step_117)
# ============================================================
print()
print("=" * 60)
print("11. DYNAMICAL MASS (step_117)")
print("=" * 60)
d = load_json(f"{OUTPUTS_DIR}/step_117_dynamical_mass_comparison.json")
if d:
    ptr = d.get("published_tension_resolution", {})
    print(f"  published_excess_dex = {ptr.get('published_excess_dex', 'N/A')}")
    print(f"  tep_reduction_dex = {ptr.get('tep_reduction_dex', 'N/A')}")
    print(f"  resolved = {ptr.get('resolved', 'N/A')}")
    olb = d.get("object_level_beta_bootstrap", {})
    print(f"  beta median = {olb.get('median', 'N/A')}")
    print(f"  beta CI = {olb.get('ci_95', 'N/A')}")

# ============================================================
# 12. Verify random-effects meta (step_091)
# ============================================================
print()
print("=" * 60)
print("12. RANDOM-EFFECTS META (step_091)")
print("=" * 60)
d = load_json(f"{OUTPUTS_DIR}/step_091_random_effects_meta_loo.json")
if d:
    s = json.dumps(d)
    for m in re.finditer(r'"(z_stat|z_combined|tau2|I2|i_squared|p_value|p_combined)"[^,]*([\d.e-]+)', s, re.IGNORECASE):
        print(f"  {m.group(1)} = {m.group(2)}")

# ============================================================
# 13. Verify MIRI robustness (step_040)
# ============================================================
print()
print("=" * 60)
print("13. MIRI ROBUSTNESS (step_040)")
print("=" * 60)
for fname in sorted(os.listdir(OUTPUTS_DIR)):
    if "040" in fname:
        d = load_json(f"{OUTPUTS_DIR}/{fname}")
        if d:
            print(f"  {fname}: {json.dumps(d, indent=2)[:1500]}")

# ============================================================
# 14. Verify cosmic SFRD (step_147, step_173)
# ============================================================
print()
print("=" * 60)
print("14. COSMIC SFRD (step_147, step_173)")
print("=" * 60)
d = load_json(f"{OUTPUTS_DIR}/step_147_cosmic_sfrd_correction.json")
if d:
    print(f"  step_147: {json.dumps(d, indent=2)[:1000]}")
d = load_json(f"{OUTPUTS_DIR}/step_173_cosmic_sfrd_correction.json")
if d:
    print(f"  step_173: {json.dumps(d, indent=2)[:1000]}")

# ============================================================
# 15. Verify Steiger Z-tests (step_138)
# ============================================================
print()
print("=" * 60)
print("15. STEIGER Z-TESTS (step_138)")
print("=" * 60)
d = load_json(f"{OUTPUTS_DIR}/step_138_environmental_screening_steiger.json")
if d:
    print(f"  step_138: {json.dumps(d, indent=2)[:2000]}")

# ============================================================
# 16. Verify κ_gal (step_123)
# ============================================================
print()
print("=" * 60)
print("16. κ_gal ERROR BUDGET (step_123)")
print("=" * 60)
d = load_json(f"{OUTPUTS_DIR}/step_123_kappa_gal_error_budget.json")
if d:
    print(f"  keys: {list(d.keys())}")
    s = json.dumps(d, indent=2)
    print(s[:1500])

print()
print("=" * 60)
print("AUDIT COMPLETE")
print("=" * 60)
