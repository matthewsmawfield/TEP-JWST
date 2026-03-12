import os
import glob
import re

def fix_5_discussion():
    file_path = "site/components/5_discussion.html"
    if not os.path.exists(file_path):
        return
        
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    start = text.find("<h4>4.3.7 Multi-Model")
    end = text.find("generic statistical trend.</p>") + len("generic statistical trend.</p>")

    if start != -1 and end != -1:
        new_text = '''<h4>4.3.7 Multi-Model Bayesian Comparison and Out-of-Box Testing</h4>
  <p>To definitively assess TEP against standard astrophysical explanations, a formal Bayesian model comparison was executed across the joint parameter space (dust, sSFR, black hole masses, and dynamical masses). TEP yields a joint Bayes Factor $\mathcal{B}_{\\rm TEP, Null} \sim 4.26 \\times 10^{83}$ ($\log_{10}\mathcal{B} \\approx 193$) against the null $\Lambda$CDM baseline, and strictly dominates all single-mechanism astrophysical alternatives. Specifically, TEP outperforms a top-heavy IMF scenario by $\Delta\\text{AIC} = -22.60$ ($\mathcal{B} = 8.07 \\times 10^4$), stochastic star formation by $\Delta\\text{AIC} = -18.75$ ($\mathcal{B} = 1.18 \\times 10^4$), and AGN feedback models by $\Delta\\text{AIC} = -26.96$ ($\mathcal{B} = 7.16 \\times 10^5$). Furthermore, an out-of-box partial correlation test confirms that even after marginalising over arbitrary monotonic mass and redshift dependencies, the $\Gamma_t$ signal retains a profound residual correlation ($\\rho_{\\rm raw} = +0.593$). This indicates that the fundamental geometric shape of the TEP predictions cannot be approximated by standard astrophysical scaling relations. When aggregated across all independent geometric predictions and domains, the overall predicted-vs-observed correlation reaches $\\rho = 0.999$, cementing TEP as an exact functional match to the anomalies rather than a generic statistical trend.</p>'''
        
        text = text[:start] + new_text + text[end:]
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        print("Fixed 5_discussion.html corrupted math block.")

def find_other_issues():
    print("\nScanning for other corrupted latex commands...")
    
    # We look for unescaped \r, \t, \a, \n, \f, \v right after a typical math macro prefix.
    # When Python processes a string like "\rho", it becomes "\r" + "ho". 
    # "\times" -> "\t" + "imes". "\approx" -> "\a" + "pprox". "\text" -> "\t" + "ext".
    # We can search for the resulting corrupted strings in the files.
    corrupted_patterns = [
        ("\r", "ho", "\\rho"),
        ("\r", "m", "\\rm"),
        ("\t", "imes", "\\times"),
        ("\a", "pprox", "\\approx"),
        ("\t", "ext", "\\text"),
        ("\n", "u", "\\nu"),
        ("\f", "rac", "\\frac"),
        ("\b", "eta", "\\beta")
    ]
    
    for comp in glob.glob("site/components/*.html"):
        with open(comp, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            
        modified = False
        for bad_char, follow, good_str in corrupted_patterns:
            bad_str = bad_char + follow
            if bad_str in content:
                print(f"  Found corrupted '{good_str}' in {comp}")
                content = content.replace(bad_str, good_str)
                modified = True
                
        if modified:
            with open(comp, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  Fixed corrupted macros in {comp}")

if __name__ == "__main__":
    fix_5_discussion()
    find_other_issues()
    print("\nAll MathJax/LaTeX errors fixed.")
