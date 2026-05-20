"""
generate_data.py
Generates synthetic HR dataset → data/hr_data.csv
Run: python generate_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate(n: int = 2500, seed: int = 7) -> pd.DataFrame:
    np.random.seed(seed)

    depts   = ["Engineering","Product","Sales","Marketing",
               "Finance","HR","Operations","Customer Success"]
    roles   = ["Analyst","Senior Analyst","Manager",
               "Senior Manager","Director","VP","Individual Contributor"]
    bands   = ["L1","L2","L3","L4","L5","L6"]
    exits   = ["Resignation","Termination","Retirement","Contract End","Transfer"]
    cities  = ["Bangalore","Mumbai","Hyderabad","Pune","Delhi","Chennai","Kolkata","Noida"]
    ed_lvl  = ["Bachelor's","Master's","MBA","PhD","Diploma"]

    df = pd.DataFrame({
        "emp_id":        [f"EMP{str(i).zfill(5)}" for i in range(n)],
        "department":    np.random.choice(depts, n, p=[.22,.12,.18,.10,.08,.06,.14,.10]),
        "role":          np.random.choice(roles, n),
        "band":          np.random.choice(bands, n, p=[.18,.22,.25,.18,.10,.07]),
        "gender":        np.random.choice(["Male","Female","Non-Binary"], n, p=[.54,.43,.03]),
        "city":          np.random.choice(cities, n),
        "education":     np.random.choice(ed_lvl, n, p=[.35,.25,.25,.08,.07]),
        "age":           np.random.randint(22, 58, n),
        "tenure_years":  np.round(np.random.exponential(3.5, n).clip(0.1, 20), 1),
        "salary":        np.round(np.random.lognormal(11.2, 0.45, n), -2),
        "performance":   np.round(np.random.beta(5, 2, n) * 4 + 1, 1).clip(1, 5),
        "satisfaction":  np.round(np.random.beta(4, 2, n) * 4 + 1, 1).clip(1, 5),
        "engagement":    np.round(np.random.beta(3, 2, n) * 4 + 1, 1).clip(1, 5),
        "leaves_taken":  np.random.randint(0, 32, n),
        "trainings":     np.random.randint(0, 9, n),
        "promotions":    np.random.randint(0, 4, n),
        "attrition":     np.random.choice([0, 1], n, p=[.82, .18]),
        "exit_reason":   np.random.choice(exits, n),
        "hire_year":     np.random.randint(2015, 2024, n),
        "remote_pct":    np.random.choice([0, 25, 50, 75, 100], n, p=[.15,.10,.30,.20,.25]),
        "overtime_hrs":  np.random.randint(0, 25, n),
        "manager_rating":np.round(np.random.beta(4, 2, n) * 4 + 1, 1).clip(1, 5),
        "peer_rating":   np.round(np.random.beta(4, 2, n) * 4 + 1, 1).clip(1, 5),
        "skip_meetings_pct": np.round(np.random.uniform(0, 0.4, n), 2),
    })

    # Composite attrition risk score (weighted formula)
    df["attrition_risk"] = np.round(
        0.30 * (1 - df["satisfaction"] / 5)
      + 0.25 * (1 - df["engagement"] / 5)
      + 0.20 * (df["overtime_hrs"] / 24)
      + 0.15 * (1 - df["tenure_years"] / 20)
      + 0.10 * np.random.rand(n), 3
    ).clip(0, 1)

    df["risk_band"] = pd.cut(
        df["attrition_risk"],
        bins=[0, 0.33, 0.66, 1],
        labels=["Low","Medium","High"]
    )

    df["cost_to_replace"]  = np.round(df["salary"] * np.random.uniform(0.5, 2.0, n), -2)
    df["days_since_promo"] = np.random.randint(0, 1500, n)
    df["target_achieved"]  = np.round(np.random.beta(5, 3, n), 2)  # 0-1

    return df


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    df = generate(n=2500)
    df.to_csv("data/hr_data.csv", index=False)
    print(f"✅  Generated {len(df):,} rows  →  data/hr_data.csv")
    print(f"    Attrition rate    : {df['attrition'].mean():.1%}")
    print(f"    High-risk employees: {(df['risk_band']=='High').sum()}")
    print(f"    Avg salary        : ₹{df['salary'].mean():,.0f}")
    print(df.dtypes.to_string())