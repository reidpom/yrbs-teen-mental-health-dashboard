import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import setup, caveat, table, test_risk, style, BLUE, ORANGE, NAVY, MUTED

setup("Model scoreboard", "🏆")
st.title("🏆 Model scoreboard")
st.write("Five models tuned on 2023 with grouped cross-validation, then tested once on 2025. The simplest one is as good as any.")

ml = table("model_ladder.csv")
fig = go.Figure()
fig.add_bar(name="2023 (cross-validated)", x=ml.model, y=ml.cv_auroc_2023, marker_color="#B9C9DE", text=ml.cv_auroc_2023.round(3), textposition="outside")
fig.add_bar(name="2025 (new students)", x=ml.model, y=ml.test_auroc_2025, marker_color=BLUE, text=ml.test_auroc_2025.round(3), textposition="outside")
fig.update_yaxes(range=[0.5, 0.9], title="Prediction score (AUROC)")
fig.update_layout(barmode="group", title="Five models: scores on 2023 and on 2025")
st.plotly_chart(style(fig, 400), width="stretch")
st.caption("AUROC: the chance the model ranks a randomly picked sad student above a randomly picked not-sad one. 0.5 is a coin flip.")

c1, c2 = st.columns(2)
with c1:
    sm = table("stronger_models.csv")
    fig = go.Figure(go.Scatter(x=sm.test_2025, y=sm.model, mode="markers", marker=dict(size=11, color=NAVY),
                               error_x=dict(type="data", symmetric=False, array=sm.hi - sm.test_2025, arrayminus=sm.test_2025 - sm.lo, color=MUTED)))
    fig.update_xaxes(title="2025 AUROC with 95% range", range=[0.79, 0.83])
    fig.update_layout(title="Can a stronger model do better? Barely.")
    st.plotly_chart(style(fig, 360), width="stretch")
    st.caption("Gradient boosting adds about 0.002; averaging it with logistic regression adds about 0.004. Ranges overlap.")
with c2:
    r = test_risk().copy()
    r["tenth"] = pd.qcut(r.risk.rank(method="first"), 10, labels=range(1, 11)).astype(int)
    dec = r.groupby("tenth").sad.mean()
    fig = go.Figure(go.Bar(x=dec.index, y=dec.values, marker_color=[ORANGE if i == 10 else BLUE for i in dec.index],
                           text=[f"{v:.0%}" for v in dec.values], textposition="outside"))
    fig.update_xaxes(title="2025 students in ten equal groups, lowest to highest predicted risk", dtick=1)
    fig.update_yaxes(tickformat=".0%", range=[0, 1], title="Actually felt sad or hopeless")
    fig.update_layout(title="A score of 0.81 in practice")
    st.plotly_chart(style(fig, 360), width="stretch")

st.subheader("Which questions matter most")
shap = table("shap_importance.csv")
n = st.slider("Questions to show", 5, 30, 12)
mdl = st.radio("Explain which model", ["Gradient boosting", "Logistic regression"], horizontal=True)
col, rk = ("shap_boosting", "rank_boosting") if mdl == "Gradient boosting" else ("shap_logistic", "rank_logistic")
top = shap.sort_values(rk).head(n).iloc[::-1]
fig = go.Figure(go.Bar(x=top[col], y=top.label, orientation="h", marker_color=[ORANGE if q == "Q80" else BLUE for q in top.question],
                       text=top[col].round(2), textposition="outside"))
r80 = shap.loc[shap.question == "Q80", rk].iloc[0]
fig.update_layout(title=f"Average push on predicted risk (mean |SHAP|) · social media ranks {r80} of {len(shap)}")
st.plotly_chart(style(fig, max(320, 26 * n)), width="stretch")

rfe = table("rfe_curve.csv")
fig = go.Figure()
fig.add_scatter(x=rfe.questions, y=rfe.cv_auroc_2023, name="2023 (cross-validated)", line=dict(color="#9BB5D6"))
fig.add_scatter(x=rfe.questions, y=rfe.test_auroc_2025, name="2025", line=dict(color=BLUE, width=3))
fig.update_xaxes(title="Number of questions kept (RFE)"); fig.update_yaxes(title="AUROC")
fig.update_layout(title="Ten questions predict almost as well as all 86")
st.plotly_chart(style(fig, 340), width="stretch")
caveat("RFE, LASSO and SHAP (on two different kinds of model) all point to the same short list.")
