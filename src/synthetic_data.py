
import numpy as np
import pandas as pd

def generate_synthetic_patients(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    ages = np.random.normal(loc=68, scale=10, size=n_samples).clip(40, 90)
    tumor_size = np.random.lognormal(mean=1.5, sigma=0.6, size=n_samples)  # cm
    ca19_9 = np.random.lognormal(mean=5, sigma=1.2, size=n_samples)
    stages = np.random.choice(['I', 'II', 'III', 'IV'], size=n_samples, p=[0.1, 0.2, 0.35, 0.35])
    treatment = np.random.choice(['Surgery', 'Chemo', 'Radio', 'Palliative'], size=n_samples,
                                 p=[0.25, 0.45, 0.15, 0.15])

    survival = []
    for s in stages:
        base = np.random.exponential(scale=18)
        factor = {'I':1.2,'II':0.9,'III':0.6,'IV':0.35}[s]
        surv = max(1, np.round(base * factor + np.random.normal(0, 2),1))
        survival.append(surv)

    df = pd.DataFrame({
        'Age': np.round(ages,1),
        'TumorSize_cm': np.round(tumor_size,2),
        'CA19_9': np.round(ca19_9,1),
        'Stage': stages,
        'TreatmentType': treatment,
        'SurvivalMonths': survival
    })
    return df

if __name__ == "__main__":
    df = generate_synthetic_patients(500)
    df.to_csv("../data/synthetic_data.csv", index=False)
    print("Synthetic dataset saved to data/synthetic_data.csv")
