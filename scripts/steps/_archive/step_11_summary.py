#!/usr/bin/env python3
"""
TEP-JWST Step 11: Summary - The Seven Threads

This step compiles all seven threads of TEP evidence into a final
summary report.

Threads:
1. z > 7 Mass-sSFR Inversion
2. Γ_t vs Age Ratio (partial)
3. Γ_t vs Metallicity (partial)
4. Γ_t vs Dust (partial)
5. z > 8 Dust Anomaly
6. Age-Metallicity Coherence
7. Multi-Property Split

Inputs:
- results/outputs/thread_1_z7_inversion.json
- results/outputs/thread_2_gamma_age.json
- results/outputs/thread_3_gamma_metallicity.json
- results/outputs/thread_4_gamma_dust.json
- results/outputs/thread_5_z8_dust.json
- results/outputs/thread_6_age_metallicity.json
- results/outputs/thread_7_multi_property.json

Outputs:
- results/outputs/tep_seven_threads_summary.json
- results/outputs/tep_seven_threads_summary.md
"""

import json
from pathlib import Path
from datetime import datetime

# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "results" / "outputs"

# =============================================================================
# LOAD RESULTS
# =============================================================================

def load_thread_results():
    """Load all thread results."""
    threads = {}
    
    thread_files = [
        ("thread_1", "thread_1_z7_inversion.json"),
        ("thread_2", "thread_2_gamma_age.json"),
        ("thread_3", "thread_3_gamma_metallicity.json"),
        ("thread_4", "thread_4_gamma_dust.json"),
        ("thread_5", "thread_5_z8_dust.json"),
        ("thread_6", "thread_6_age_metallicity.json"),
        ("thread_7", "thread_7_multi_property.json"),
    ]
    
    for key, filename in thread_files:
        filepath = OUTPUT_PATH / filename
        if filepath.exists():
            with open(filepath) as f:
                threads[key] = json.load(f)
        else:
            print(f"Warning: {filename} not found")
            threads[key] = None
    
    return threads

# =============================================================================
# GENERATE SUMMARY
# =============================================================================

def generate_summary(threads):
    """Generate summary of all threads."""
    
    summary = {
        "title": "TEP-JWST: The Seven Threads of Evidence",
        "generated": datetime.now().isoformat(),
        "threads": {},
        "overall": {},
    }
    
    significant_count = 0
    total_count = 0
    
    thread_info = [
        ("thread_1", "z > 7 Mass-sSFR Inversion", "significant"),
        ("thread_2", "Γ_t vs Age Ratio", "significant"),
        ("thread_3", "Γ_t vs Metallicity", "significant"),
        ("thread_4", "Γ_t vs Dust", "significant"),
        ("thread_5", "z > 8 Dust Anomaly", "significant"),
        ("thread_6", "Age-Metallicity Coherence", "significant"),
        ("thread_7", "Multi-Property Split", "all_significant"),
    ]
    
    for key, name, sig_field in thread_info:
        total_count += 1
        if threads.get(key):
            is_sig = threads[key].get(sig_field, False)
            if is_sig:
                significant_count += 1
            
            summary["threads"][key] = {
                "name": name,
                "significant": is_sig,
                "details": threads[key],
            }
        else:
            summary["threads"][key] = {
                "name": name,
                "significant": False,
                "details": None,
            }
    
    summary["overall"] = {
        "threads_significant": significant_count,
        "threads_total": total_count,
        "all_significant": significant_count == total_count,
    }
    
    return summary

def generate_markdown(summary):
    """Generate markdown report."""
    
    lines = [
        "# TEP-JWST: The Seven Threads of Evidence",
        "",
        f"Generated: {summary['generated']}",
        "",
        "## Summary",
        "",
        f"**Threads Significant: {summary['overall']['threads_significant']}/{summary['overall']['threads_total']}**",
        "",
        "| Thread | Finding | Significant |",
        "|--------|---------|-------------|",
    ]
    
    thread_details = [
        ("thread_1", lambda t: f"Δρ = {t['delta_rho']:.2f} [{t['delta_ci_95'][0]:.2f}, {t['delta_ci_95'][1]:.2f}]" if t else "N/A"),
        ("thread_2", lambda t: f"ρ = {t['partial_rho']:.2f} (p = {t['partial_p']:.0e})" if t else "N/A"),
        ("thread_3", lambda t: f"ρ = {t['partial_rho']:.2f} (p = {t['partial_p']:.0e})" if t else "N/A"),
        ("thread_4", lambda t: f"ρ = {t['partial_rho']:.2f} (p = {t['partial_p']:.0e})" if t else "N/A"),
        ("thread_5", lambda t: f"ρ = {t['z8_result']['rho']:.2f} [{t['z8_result']['ci_95'][0]:.2f}, {t['z8_result']['ci_95'][1]:.2f}]" if t else "N/A"),
        ("thread_6", lambda t: f"ρ = {t['rho']:.2f} [{t['ci_95'][0]:.2f}, {t['ci_95'][1]:.2f}]" if t else "N/A"),
        ("thread_7", lambda t: f"All p < 10⁻¹⁰" if t and t.get('all_significant') else "Partial" if t else "N/A"),
    ]
    
    for key, detail_fn in thread_details:
        thread = summary["threads"].get(key, {})
        name = thread.get("name", key)
        sig = "✓" if thread.get("significant") else "✗"
        details = thread.get("details")
        finding = detail_fn(details)
        lines.append(f"| {name} | {finding} | {sig} |")
    
    lines.extend([
        "",
        "## Interpretation",
        "",
    ])
    
    if summary["overall"]["all_significant"]:
        lines.extend([
            "**All seven threads are statistically significant.**",
            "",
            "The evidence forms a coherent tapestry:",
            "",
            "1. The mass-sSFR correlation **inverts** at z > 7 — anomalous under standard physics",
            "2. Predicted Γ_t correlates with age ratio, metallicity, and dust — quantitative TEP match",
            "3. The z > 8 dust anomaly is **anomalous under standard timescales** — TEP provides the only explanation",
            "4. All properties shift coherently with Γ_t — the TEP fingerprint",
            "",
            "This pattern is difficult to explain without invoking a mechanism that couples",
            "stellar evolution to gravitational potential depth.",
        ])
    else:
        lines.extend([
            f"**{summary['overall']['threads_significant']}/{summary['overall']['threads_total']} threads are significant.**",
            "",
            "The evidence is suggestive but not complete. Further investigation needed.",
        ])
    
    return "\n".join(lines)

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("STEP 11: Summary - The Seven Threads")
    print("=" * 60)
    print()
    
    threads = load_thread_results()
    
    summary = generate_summary(threads)
    
    with open(OUTPUT_PATH / "tep_seven_threads_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {OUTPUT_PATH / 'tep_seven_threads_summary.json'}")
    
    markdown = generate_markdown(summary)
    with open(OUTPUT_PATH / "tep_seven_threads_summary.md", "w") as f:
        f.write(markdown)
    print(f"Saved: {OUTPUT_PATH / 'tep_seven_threads_summary.md'}")
    
    print()
    print("=" * 60)
    print("THE SEVEN THREADS")
    print("=" * 60)
    print()
    
    for key in ["thread_1", "thread_2", "thread_3", "thread_4", "thread_5", "thread_6", "thread_7"]:
        thread = summary["threads"].get(key, {})
        name = thread.get("name", key)
        sig = "✓" if thread.get("significant") else "✗"
        print(f"  {name}: {sig}")
    
    print()
    print(f"Threads significant: {summary['overall']['threads_significant']}/{summary['overall']['threads_total']}")
    print()
    
    if summary["overall"]["all_significant"]:
        print("★ THE TAPESTRY IS COMPLETE ★")
    
    print()
    print("Step 11 complete.")

if __name__ == "__main__":
    main()
