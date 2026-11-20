import numpy as np
import plotly.graph_objects as go
import streamlit as st
from common import setup, caveat, test_risk, style, BLUE, ORANGE, MUTED

setup("The fairness slider", "⚖️")
st.title("⚖️ The fairness slider")
st.write("Imagine a school uses the model to decide which students a counselor checks in with. Students above the cutoff get flagged. "
         "Move the cutoff and see who gets caught. Girls report sadness about twice as often as boys, and that changes what one cutoff does.")

r = test_risk()
mode = st.radio("Cutoff rule", ["One cutoff for everyone", "Separate cutoffs for girls and boys"], horizontal=True)


def stats(sub, cut):
    flag = sub.risk >= cut; sad = sub.sad == 1
    return {"caught": (flag & sad).sum() / sad.sum(), "false_alarm": (flag & ~sad).sum() / (~sad).sum(),
            "flagged": flag.mean(), "precision": (flag & sad).sum() / max(flag.sum(), 1), "n": len(sub)}


if mode == "One cutoff for everyone":
    cut = st.slider("Risk cutoff", 0.05, 0.95, 0.50, 0.01)
    cuts = {"Girl": cut, "Boy": cut}
else:
    a, b = st.columns(2)
    cuts = {"Girl": a.slider("Cutoff for girls", 0.05, 0.95, 0.60, 0.01), "Boy": b.slider("Cutoff for boys", 0.05, 0.95, 0.34, 0.01)}
    st.caption("Starting values give both groups about the same catch rate, close to the separate-cutoff result in our report (68% vs 70%).")

res = {sex: stats(r[r.sex == sex], cuts[sex]) for sex in ["Girl", "Boy"]}
c = st.columns(4)
c[0].metric("Sad girls caught", f"{res['Girl']['caught']:.0%}")
c[1].metric("Sad boys caught", f"{res['Boy']['caught']:.0%}", f"{(res['Boy']['caught'] - res['Girl']['caught']) * 100:+.0f} points vs girls", delta_color="off", delta_arrow="off")
c[2].metric("False alarms, girls", f"{res['Girl']['false_alarm']:.0%}", "of girls who were not sad", delta_color="off", delta_arrow="off")
c[3].metric("False alarms, boys", f"{res['Boy']['false_alarm']:.0%}", "of boys who were not sad", delta_color="off", delta_arrow="off")

fig = go.Figure()
for metric, label in [("caught", "Sad students caught"), ("false_alarm", "Not-sad students flagged (false alarms)")]:
    fig.add_bar(name=label, x=["Girls", "Boys"], y=[res["Girl"][metric], res["Boy"][metric]],
                marker_color=BLUE if metric == "caught" else ORANGE, text=[f"{res[s][metric]:.0%}" for s in ["Girl", "Boy"]], textposition="outside")
fig.update_yaxes(tickformat=".0%", range=[0, 1.1])
fig.update_layout(barmode="group", title="At the chosen cutoff(s)")
st.plotly_chart(style(fig, 380), width="stretch")

# trade-off curve
grid = np.linspace(0.05, 0.95, 91)
fig2 = go.Figure()
for sex, col in [("Girl", "#B8322B"), ("Boy", BLUE)]:
    sub = r[r.sex == sex]
    fig2.add_scatter(x=grid, y=[stats(sub, g)["caught"] for g in grid], name=f"{sex}s: sad students caught", line=dict(color=col, width=2.5))
    fig2.add_scatter(x=[cuts[sex]], y=[res[sex]["caught"]], mode="markers", marker=dict(size=12, color=col), showlegend=False)
fig2.update_xaxes(title="Risk cutoff"); fig2.update_yaxes(title="Share of sad students caught", tickformat=".0%")
fig2.update_layout(title="Raising the cutoff misses boys first")
st.plotly_chart(style(fig2, 360), width="stretch")

with st.expander("Why does this happen?"):
    st.markdown("""
The model ranks girls and boys almost equally well (2025 AUROC 0.80 for girls, 0.79 for boys). But girls report sadness far more often,
so the model's risk scores for girls sit higher overall. One cutoff therefore flags many more girls, and catches fewer sad boys.
Separate cutoffs trade a few more false alarms among boys for catching sad boys as often as sad girls.
This is a screening illustration on survey data, not a diagnostic tool.
""")
caveat("Risk scores come from the logistic regression trained on 2023, applied to 9,686 students in 2025.")
