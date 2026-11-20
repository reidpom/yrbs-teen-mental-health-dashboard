import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import setup, caveat, table, students, style, BLUE, ORANGE, MUTED, SOCIAL_ORDER

setup("The social media story", "📱")
st.title("📱 The social media story")
st.write("Heavy social media users are sadder. But heavy use goes along with less sleep, more bullying and more of everything else. "
         "Add what we know about a student one step at a time and watch what happens to the social media link.")

s = students()
# ---- raw picture first
st.subheader("Step 0: the raw picture")
yr = st.radio("Survey year", [2023, 2025], horizontal=True, key="yr")
g = s[s.year == yr].groupby("social").apply(lambda x: pd.Series({"rate": np.average(x.sad, weights=x.weight), "n": len(x)}), include_groups=False)
g = g.reindex(SOCIAL_ORDER)
fig = go.Figure(go.Bar(x=g.index, y=g.rate, marker_color=[ORANGE if i == len(g) - 1 else BLUE for i in range(len(g))],
                       text=[f"{v:.0%}" for v in g.rate], textposition="outside", customdata=g.n,
                       hovertemplate="%{x}<br>%{y:.1%} sad or hopeless<br>%{customdata:,} students<extra></extra>"))
fig.update_yaxes(tickformat=".0%", range=[0, max(g.rate) * 1.2], title="Felt sad or hopeless (weighted)")
st.plotly_chart(style(fig, 380), width="stretch")
st.caption("How often students use social media, and the share who felt sad or hopeless. Survey-weighted to represent U.S. high school students.")

# ---- the staged models
st.subheader("Steps 1–3: add what else we know")
c1, c2 = st.columns(2)
outcome = c1.radio("Outcome", ["Sad or hopeless (Q26)", "Poor mental health, past 30 days (Q84)"], horizontal=False)
weighted = c2.toggle("Use survey weights (re-check)", value=False)
t = table("staged_social_media.csv")
t = t[(t.outcome.str.startswith(outcome[:10])) & (t.weighted == weighted)].reset_index(drop=True)
steps = {1: "Background + social media", 2: "+ sleep and bullying", 3: "+ everything else"}
step = st.select_slider("How much we know about each student", options=[1, 2, 3], value=1, format_func=lambda k: steps[k])

o = t[t.odds.notna()].reset_index(drop=True).iloc[:step]
left, right = st.columns([3, 2])
with left:
    fig = go.Figure(go.Bar(x=[steps[i + 1] for i in range(len(o))], y=o.odds, marker_color=[BLUE] * (len(o) - 1) + [ORANGE],
                           error_y=dict(type="data", symmetric=False, array=o.hi - o.odds, arrayminus=o.odds - o.lo, color=MUTED, thickness=1.5),
                           text=[f"{v:.2f}×" for v in o.odds], textposition="inside", insidetextanchor="start", textfont=dict(color="white", size=16),
                           hovertemplate="%{x}<br>odds ratio %{y:.2f}<extra></extra>"))
    fig.add_hline(y=1, line_dash="dash", line_color=MUTED, annotation_text="1.0 = no link", annotation_position="bottom right")
    fig.update_yaxes(range=[0, 3.0 if outcome.startswith("Sad") else 4.5], title="Odds of the outcome vs non-users")
    fig.update_xaxes(range=[-0.5, 2.5], categoryorder="array", categoryarray=list(steps.values()))
    fig.update_layout(title="Heaviest users (more than once an hour) vs non-users")
    st.plotly_chart(style(fig, 420), width="stretch")
with right:
    cur = o.iloc[-1]
    st.metric("Odds ratio, heaviest users", f"{cur.odds:.2f}×", f"range {cur.lo:.2f} to {cur.hi:.2f}", delta_color="off", delta_arrow="off")
    sc = t.set_index("stage").test_2025
    st.metric("2025 prediction score", f"{sc.iloc[step]:.3f}", f"{sc.iloc[step] - sc.iloc[0]:+.3f} vs background only", delta_color="off", delta_arrow="off")
    if step == 1:
        st.info("Social media alone adds only about 0.01 to the prediction score.")
    elif step == 2:
        st.info("Sleep and bullying add far more to prediction than social media did, yet the social media link barely moves.")
    else:
        st.warning("With everything known, the link shrinks. Its range now dips close to 1.0 (in the re-checks it crosses 1.0), so it is small and uncertain.")

with st.expander("How to read this"):
    st.markdown("""
- **Odds ratio:** 1.74× means the heaviest users' odds of feeling sad are 1.74 times the odds of students who don't use social media, among students who are otherwise alike on what's in the model.
- **The range** comes from re-running the model 300 times on resampled schools. A range that includes 1.0 means we cannot rule out chance.
- **The prediction score** (AUROC) is how well the model ranks a sad student above a not-sad one, from 0.5 (coin flip) to 1.0.
- The heaviest users are 32% of students; students on about once an hour show a smaller gap (1.35×).
""")
caveat("The odds come from logistic regressions trained on 2023; scores are measured on 2025.")
