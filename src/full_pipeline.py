import pandas as pd, matplotlib.pyplot as plt, joblib, json, os, sys
from pathlib import Path
from synthetic_data import generate_synthetic_patients
from train_models import train_all_models
# --- Step 1: data ---------------------------------
Path("../data").mkdir(exist_ok=True, parents=True)
df = generate_synthetic_patients(5000)                 # original
df.to_csv("../data/synthetic_data.csv", index=False)

# TODO:  call your VAE here → df_aug
df_aug = df.copy()                                    # stub: no VAE yet

# --- Step 2: models --------------------------------
perf_orig = train_all_models(df)
perf_aug  = train_all_models(df_aug)
perf = perf_orig.join(perf_aug, lsuffix='_orig', rsuffix='_vae')

Path("../results").mkdir(exist_ok=True, parents=True)
perf.to_csv("../results/performance.csv")

# --- Step 3: plot ----------------------------------
ax = perf['MAE_orig'].plot(kind='bar', label='Orig', alpha=.7)
perf['MAE_vae'].plot(kind='bar', label='VAE', alpha=.7, ax=ax, figsize=(8,4))
ax.set_ylabel("MAE  (months)"); ax.set_title("Model comparison")
plt.legend(); plt.tight_layout(); plt.savefig("../results/performance_bar.png")
plt.close()

# --- Step 4: push into PPT -------------------------
from pptx import Presentation
prs = Presentation("../presentation/Group26_Presentation.pptx")
MAE_best = perf['MAE_vae'].min()
improve  = 100*(perf['MAE_orig'].min() - MAE_best)/perf['MAE_orig'].min()

replace_map = {
    "«Nreal»": str(len(df)),
    "«Nsyn»" : str(len(df_aug)-len(df)),
    "«rows × cols»": f"{df.shape[0]} × {df.shape[1]}",
    "«k»": "0",
    "«x»": f"{improve:.1f}",
    "«y»": f"{(perf.loc['ElasticNet','MAE_orig']-perf.loc['ElasticNet','MAE_vae']):.2f}",
    "«Δ»": f"{improve:.1f}",
    "«overall %»": f"{improve:.1f}",
}
for slide in prs.slides:
    for shape in slide.shapes:
        if not shape.has_text_frame: continue
        for p in shape.text_frame.paragraphs:
            for run in p.runs:
                for key,val in replace_map.items():
                    if key in run.text: run.text = run.text.replace(key,val)
prs.save("../presentation/Group26.pptx")
print("✓ Pipeline finished — see results/ and updated PPTX")
