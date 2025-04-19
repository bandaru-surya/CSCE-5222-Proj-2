import pandas as pd, numpy as np, matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
import joblib, warnings

BASE        = Path(__file__).resolve().parent.parent
DATA_CSV    = BASE / "data"   / "synthetic_data.csv"
AUG_CSV     = BASE / "data"   / "synthetic_data.csv"   # TODO change to VAE csv later
PERF_CSV    = BASE / "results"/ "performance.csv"      # from run_full_pipeline
MODEL_DIR   = BASE / "models"
RESULTS_DIR = BASE / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# ---------- 1. histogram of survival ---------------------------
df = pd.read_csv(DATA_CSV)
plt.figure(figsize=(4,3))
sns.histplot(df["SurvivalMonths"], bins=20, kde=False, color="#4c8eda")
plt.title("Distribution of Survival Months")
plt.xlabel("Months"); plt.ylabel("Count"); plt.tight_layout()
plt.savefig(RESULTS_DIR / "hist_survival.png"); plt.close()

# ---------- 2. correlation heat‑map ----------------------------
num_cols = df.select_dtypes(include=["float64","int64"]).columns
corr = df[num_cols].corr()
plt.figure(figsize=(5,4))
sns.heatmap(corr, cmap="vlag", annot=False, linewidths=.5)
plt.title("Feature Correlation Heat‑map"); plt.tight_layout()
plt.savefig(RESULTS_DIR / "corr_heatmap.png"); plt.close()

# ---------- 3. feature count bar -------------------------------
# assume correlation filter dropped some columns & we saved that list
try:
    drops = np.loadtxt(BASE/"results/to_drop.txt", dtype=str)
except FileNotFoundError:
    drops = []    # fallback
before = len(num_cols)
after  = before - len(drops)
plt.figure(figsize=(3,3))
sns.barplot(x=["Before","After"], y=[before, after], palette="Blues")
plt.title("Feature count after FE"); plt.ylabel("# features"); plt.tight_layout()
plt.savefig(RESULTS_DIR / "feature_count.png"); plt.close()

# ---------- 4. t‑SNE plot real vs synthetic --------------------
df_aug = pd.read_csv(AUG_CSV)
df_aug["label"] = "Synthetic"
df["label"]    = "Original"
combo = pd.concat([df, df_aug]).sample(frac=1, random_state=42).reset_index(drop=True)
X = combo.select_dtypes(include=["float64","int64"]).drop(columns="SurvivalMonths")

warnings.filterwarnings("ignore", category=FutureWarning)
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
coords = tsne.fit_transform(X)
plt.figure(figsize=(5,4))
sns.scatterplot(x=coords[:,0], y=coords[:,1], hue=combo["label"],
                alpha=.6, palette={"Original":"#4c8eda","Synthetic":"#f98e52"})
plt.title("t‑SNE 2‑D Projection"); plt.legend(); plt.tight_layout()
plt.savefig(RESULTS_DIR / "tsne_plot.png"); plt.close()

# ---------- 5. MAE bar already exists (skip) -------------------

# ---------- 6. feature‑importance (best tree) ------------------
# pick LightGBM if present else XGBoost
best = None
for mdl in ["LightGBM", "XGBoost"]:
    pkl = MODEL_DIR / f"{mdl}_model.pkl"
    if pkl.exists(): 
        best = (mdl, joblib.load(pkl)); break

if best:
    mdl_name, pipe = best
    inner = pipe.named_steps["model"]
    if hasattr(inner, "feature_importances_"):
        pre  = pipe.named_steps.get("prep") or pipe.named_steps.get("pre")
        num  = list(pre.transformers_[0][2])
        cat  = list(pre.named_transformers_["cat"].get_feature_names_out())
        fi   = pd.Series(inner.feature_importances_, index=num+cat)
        top  = fi.sort_values(ascending=True).tail(10)
        plt.figure(figsize=(5,3))
        top.plot(kind="barh", color="#4c8eda")
        plt.title(f"Top 10 Feature Importance – {mdl_name}")
        plt.tight_layout(); plt.savefig(RESULTS_DIR / "fi_best.png"); plt.close()

print("✓  All figures saved to", RESULTS_DIR.relative_to(BASE))