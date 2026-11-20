import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from common import setup, caveat, students, style, KMEANS_NAMES, RISK_NAMES, WARD_NAMES, GROUP_COLORS, SOCIAL_ORDER

setup("The student map", "🗺️")
st.title("🗺️ The student map")
st.write("Each dot is a student. Students with similar answers sit close together (UMAP built from MCA themes). "
         "The groupings never saw the mental health answers, yet sadness differs sharply between them.")

s = students()
c1, c2, c3 = st.columns([2, 1, 1])
color_by = c1.radio("Color dots by", ["Profile (k-means)", "Profile (hierarchical)", "Risk-only groups", "Felt sad or hopeless", "Social media use"], horizontal=True)
year = c2.radio("Students from", ["2023", "2025", "Both"], horizontal=True)
sample = c3.toggle("Faster: show a 40% sample", value=True)

v = s if year == "Both" else s[s.year == int(year)]
if sample:
    v = v.sample(frac=0.4, random_state=0)
v = v.copy()
if color_by.startswith("Profile (k"):
    v["color"] = v.kmeans.map(KMEANS_NAMES); order = list(KMEANS_NAMES.values()); cmap = dict(zip(order, GROUP_COLORS)); key = "kmeans"; names = KMEANS_NAMES
elif color_by.startswith("Profile (h"):
    v["color"] = v.ward.map(WARD_NAMES); order = list(WARD_NAMES.values()); cmap = dict(zip(order, GROUP_COLORS)); key = "ward"; names = WARD_NAMES
elif color_by.startswith("Risk"):
    v["color"] = v.risk_group.map(RISK_NAMES); order = list(RISK_NAMES.values()); cmap = dict(zip(order, GROUP_COLORS)); key = "risk_group"; names = RISK_NAMES
elif color_by.startswith("Felt"):
    v["color"] = v.sad.map({1: "Felt sad or hopeless", 0: "Did not"}); order = ["Did not", "Felt sad or hopeless"]
    cmap = {"Did not": "#B9C9DE", "Felt sad or hopeless": "#E34948"}; key = None
else:
    v["color"] = v.social; order = SOCIAL_ORDER
    cmap = dict(zip(SOCIAL_ORDER, px.colors.sequential.Plasma_r[1:9])); key = None

fig = px.scatter(v, x="x", y="y", color="color", category_orders={"color": order}, color_discrete_map=cmap, opacity=0.55,
                 hover_data={"x": False, "y": False, "color": True, "sex": True, "grade": True, "social": True, "year": True})
fig.update_traces(marker=dict(size=3.2))
fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
fig.update_layout(legend=dict(orientation="v", itemsizing="constant", y=1, x=1.01))
st.plotly_chart(style(fig, 620), width="stretch")
st.caption("The map is a picture only: distances between far-apart regions mean little. The group numbers and sad rates below are the evidence.")

if key:
    rows = []
    for g, nm in names.items():
        r = {"Group": nm}
        for yr in (2023, 2025):
            m = (s.year == yr) & (s[key] == g); a = s.year == yr
            r[f"Share {yr}"] = m.sum() / a.sum(); r[f"Sad {yr}"] = np.average(s.sad[m], weights=s.weight[m])
        r["Heaviest social media users (2023)"] = (s.social_code[(s.year == 2023) & (s[key] == g)] == 8).mean()
        rows.append(r)
    tab = pd.DataFrame(rows)
    st.dataframe(tab.style.format({c: "{:.0%}" for c in tab.columns if c != "Group"}).background_gradient(subset=["Sad 2023", "Sad 2025"], cmap="Reds"),
                 hide_index=True, width="stretch")
    st.caption("Sad rates are survey-weighted. 2025 students were placed into the groups learned from 2023.")
    if key == "ward":
        st.info("Hierarchical clustering agrees broadly with k-means (adjusted Rand 0.66 in 2023, 0.78 in 2025). It finds the same low-risk and heavy-risk groups but draws the middle borders differently.")
    if key == "risk_group":
        st.info("Built from risk questions only, treating \"not applicable\" answers as neutral. These groups separate risk better on 2025 than the original profiles.")
caveat()
