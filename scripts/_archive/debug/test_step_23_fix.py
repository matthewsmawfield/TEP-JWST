
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

# Mock data generation to simulate the issue
np.random.seed(42)
N = 1000
log_Mhalo = np.random.uniform(10, 13, N)
z = np.random.uniform(5, 10, N)
# Create a synthetic age_ratio that partially depends on Mass (mimicking TEP) but with noise
# Assume TEP is true with alpha=0.58
M_REF = 1e11
true_alpha = 0.58
gamma_true = true_alpha * (10**log_Mhalo / M_REF)**(1/3)
# True age ratio is constant-ish, observed is boosted by gamma
base_age_ratio = np.random.normal(0.1, 0.02, N)
obs_age_ratio = base_age_ratio * gamma_true

df = pd.DataFrame({'log_Mhalo': log_Mhalo, 'age_ratio': obs_age_ratio})

def compute_gamma(alpha, data):
    M_h = 10 ** data['log_Mhalo']
    return alpha * (M_h / M_REF) ** (1/3)

def objective_std(alpha):
    gamma = compute_gamma(alpha, df)
    corrected = df['age_ratio'] / gamma
    return corrected.std()

def objective_cv(alpha):
    gamma = compute_gamma(alpha, df)
    corrected = df['age_ratio'] / gamma
    return corrected.std() / corrected.mean()

print("--- Testing Standard Deviation Objective (Current Pipeline) ---")
res_std = minimize_scalar(objective_std, bounds=(0.1, 5.0), method='bounded')
print(f"Optimal Alpha (Std): {res_std.x:.4f}")
print(f"Bound hit? {np.isclose(res_std.x, 5.0)}")

print("\n--- Testing Coefficient of Variation Objective (Proposed Fix) ---")
res_cv = minimize_scalar(objective_cv, bounds=(0.1, 5.0), method='bounded')
print(f"Optimal Alpha (CV): {res_cv.x:.4f}")
print(f"True Alpha was: {true_alpha}")
