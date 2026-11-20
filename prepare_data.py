"""Builds the small data folder the dashboard reads (dashboard/data/). Run once after the analysis scripts.
Nothing here is trained or computed inside the app; the app only reads these files.
Usage: python3 prepare_data.py <MIS545 folder>
"""
import json, re, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

H = Path(sys.argv[1]).expanduser().resolve(); C = H / "YRBS_combined"; R = C / "results"
OUT = Path(__file__).resolve().parent / "data"; (OUT / "tables").mkdir(parents=True, exist_ok=True)

d = pd.read_csv(C / "yrbs_2023_2025_shared.csv", low_memory=False)
d = d[d.Q26.notna() & d.Q80.notna()].reset_index(drop=True)
y = (d.Q26 == 1).astype(int).values; tr = (d.year == 2023).values

# ---------- labels
t = (H / "YRBS_2023/2023XXH-Formats-Program.sas").read_text(encoding="latin-1").replace("\r", "")
fm = {"Q" + n: dict(re.findall(r'"(\d+)"="([^"]*)"', b)) for n, b in re.findall(r"value \$H(\d+)[A-Z]*\n(.*?);", t, re.S)}
ql = pd.read_csv(C / "shared_columns.csv").set_index("column (2023 number)")["label"].to_dict()
PLAIN = {"Q89": "Emotional abuse by a parent", "Q2": "Sex", "Q85": "Hours of sleep", "Q103": "Close to people at school", "Q64": "Sexual identity",
         "Q24": "Bullied at school", "Q25": "Bullied online", "Q35": "Ever vaped", "Q41": "Age at first drink", "Q47": "Age first tried marijuana",
         "Q80": "Social media use", "Q90": "Physical abuse by a parent", "Q23": "Treated unfairly at school (race)", "Q63": "Sex of sexual contacts",
         "Q20": "Sexual violence", "Q67": "Trying to lose weight", "Q66": "Weight perception", "Q14": "Felt unsafe at school", "Q92": "Misused pain medicine",
         "Q99": "Basic needs met at home", "Q18": "Saw violence in neighborhood", "Q78": "Sports teams", "Q19": "Forced sex", "Q46": "Ever used marijuana",
         "Q31": "Ever smoked", "Q1": "Age", "Q3": "Grade", "Q86": "Unstable housing", "Q76": "Days active", "Q104": "Parental monitoring",
         "Q26": "Felt sad or hopeless", "Q84": "Poor mental health (past 30 days)"}
name = lambda q: PLAIN.get(q, ql.get(q, q))
ans = lambda q, v: fm.get(q, {}).get(str(int(v)), str(int(v)))
qs = [c for c in d.columns if re.fullmatch(r"Q\d+", c)]
json.dump({q: {"name": name(q), "label": ql.get(q, q), "answers": fm.get(q, {})} for q in qs}, open(OUT / "questions.json", "w"), indent=1)

# ---------- one row per student (no IDs): map, groupings, outcome, a few descriptors, weight
um = pd.read_csv(R / "umap_coords.csv"); km = pd.read_csv(R / "profiles.csv"); wd = pd.read_csv(R / "hierarchical_labels.csv")
ro = pd.read_csv(R / "risk_only_labels.csv")
assert len(um) == len(km) == len(wd) == len(ro) == len(d) and (um.year.values == d.year.values).all()
sm = {1: "Does not use", 2: "A few times a month", 3: "About once a week", 4: "A few times a week", 5: "About once a day",
      6: "Several times a day", 7: "About once an hour", 8: "More than once an hour"}
pts = pd.DataFrame({"x": um.umap1.round(3), "y": um.umap2.round(3), "year": d.year, "kmeans": km.profile, "ward": wd.ward_group,
                    "risk_group": ro.risk_group, "sad": y, "social": d.Q80.astype(int).map(sm), "social_code": d.Q80.astype(int),
                    "sex": d.Q2.map({1: "Girl", 2: "Boy"}), "grade": d.Q3.map({1: "9th", 2: "10th", 3: "11th", 4: "12th"}), "weight": d.WEIGHT.round(4)})
pts.to_csv(OUT / "students.csv", index=False)

# ---------- question explorer: weighted share and sad rate for every answer, by year
rows = []
for q in qs:
    for yr in (2023, 2025):
        m = (d.year == yr).values & d[q].notna().values
        w = d.WEIGHT.values[m]; v = d[q].values[m].astype(int)
        for a in sorted(set(v)):
            k = v == a
            rows.append({"question": q, "year": yr, "code": a, "answer": ans(q, a), "n": int(k.sum()),
                         "share": w[k].sum() / w.sum(), "sad_rate": np.average(y[m][k], weights=w[k]) if k.sum() else np.nan})
pd.DataFrame(rows).to_csv(OUT / "question_stats.csv", index=False)

# ---------- fairness slider: 2025 predicted risk from the logistic regression + sex
tp = pd.read_csv(R / "test_probabilities_2025.csv")
assert (tp.y.values == y[~tr]).all(), "test probabilities are not in the same order as the 2025 students"
pd.DataFrame({"risk": tp["Logistic regression"].round(5), "sad": tp.y, "sex": pts.sex[~tr].values}).to_csv(OUT / "test_risk_2025.csv", index=False)

# ---------- what-if explorer: a small logistic regression on the RFE top 10 + social media (trained on 2023)
rank = pd.read_csv(R / "question_ranking.csv").sort_values("rfe_rank")
WQ = list(dict.fromkeys(rank.question.head(10).tolist() + ["Q80"]))
X = d[WQ].fillna(0).astype(int)                                          # 0 = skipped
dm = pd.get_dummies(X.astype(str), prefix_sep="=").astype(float)
lr = LogisticRegression(C=0.1, max_iter=4000).fit(dm[tr], y[tr])
auc = roc_auc_score(y[~tr], lr.predict_proba(dm[~tr])[:, 1])
coef = dict(zip(dm.columns, lr.coef_[0]))
model = {"intercept": float(lr.intercept_[0]), "auroc_2025": round(float(auc), 3), "questions": []}
for q in WQ:
    lv = sorted(X[q].unique())
    cq = {int(a): float(coef.get(f"{q}={a}", 0.0)) for a in lv}
    mean_c = float(np.mean([cq[a] for a in X[q][tr]]))                   # centering = exact SHAP for a linear model
    common = int(X[q][tr].mode()[0])
    model["questions"].append({"q": q, "name": name(q), "label": ql.get(q, q), "default": common, "mean_contrib": mean_c,
                               "answers": [{"code": int(a), "text": "Skipped" if a == 0 else ans(q, a), "coef": cq[a]} for a in lv]})
model["base_logit"] = model["intercept"] + sum(x["mean_contrib"] for x in model["questions"])
json.dump(model, open(OUT / "whatif_model.json", "w"), indent=1)
print(f"what-if model: {len(WQ)} questions, 2025 AUROC {auc:.3f}")

# ---------- results tables used by the pages and the chat assistant
TABLES = {
    "staged_social_media.csv": ("robustness.csv", "Social media odds ratio (more than once an hour vs non-users) and 2025 score as information is added in stages; both outcomes, with and without survey weights. lo/hi = 95% bootstrap range."),
    "model_ladder.csv": ("model_ladder.csv", "Five tuned models: 2023 cross-validated AUROC, 2025 test AUROC, recall, precision, F1 at a 0.5 cutoff."),
    "stronger_models.csv": ("scoring_models.csv", "Gradient boosting and averaging versus logistic regression, 2025 AUROC with ranges; share sad in the top 10% by risk."),
    "shap_importance.csv": ("shap_importance.csv", "SHAP importance (mean |SHAP|, log-odds) for every question, boosting and logistic regression, with ranks."),
    "question_ranking.csv": ("question_ranking.csv", "RFE rank and LASSO weight for every question."),
    "selection_stability.csv": ("ranking_stability.csv", "Share of 40 resamples in which each question made the RFE top 10 / short LASSO list."),
    "rfe_curve.csv": ("rfe_curve.csv", "Prediction score versus number of questions kept by RFE."),
    "fairness_2025.csv": ("fairness_2025.csv", "2025 AUROC, recall and false-alarm rate by sex and race/ethnicity."),
    "cutoffs_2025.csv": ("cutoffs_2025.csv", "One cutoff versus separate cutoffs for girls and boys: share of sad students caught, false alarms."),
    "profiles_kmeans.csv": ("profile_summary.csv", "Five k-means profiles: share of students and sad rate in 2023 and 2025, plus social media use inside each."),
    "profiles_hierarchical_vs_kmeans.csv": ("hierarchical_check.csv", "Ward hierarchical clustering versus k-means: share and sad rate per group, both years."),
    "risk_only_groups.csv": ("risk_only_2025.csv", "Groups built from risk questions only: share and sad rate in 2023 and 2025."),
    "protective_factors.csv": ("protective_factors.csv", "Social media odds ratio with and without each protective factor (sleep, school closeness, activity, sports, parental monitoring)."),
    "calibration_2025.csv": ("calibration_2025.csv", "Predicted versus actual sad rate in each tenth of 2025 students by risk."),
    "single_question_scores.csv": ("single_question_scores.csv", "2025 AUROC of each question used alone."),
    "codebook_check_2025.txt": ("codebook_check_2025.txt", "Checks of the 2025 file against CDC's 2025 Data User's Guide."),
}
for out, (src, _) in TABLES.items():
    shutil.copy(R / src, OUT / "tables" / out)
json.dump({k: v[1] for k, v in TABLES.items()}, open(OUT / "tables" / "index.json", "w"), indent=1)

# the write-up, without image lines, as background for the chat assistant
src = H / "report.md"                                                  # the project write-up (Markdown)
(OUT / "report.md").write_text("\n".join(l for l in src.read_text().splitlines() if not l.startswith("![")))
print("wrote", sorted(p.name for p in OUT.iterdir()), f"{sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) / 1e6:.1f} MB")
