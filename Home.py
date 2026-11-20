import streamlit as st
from common import setup, caveat, table, students

setup("Home", "🏠")
st.title("Social media and teen mental health")
st.markdown("#### What nearly 25,000 high school students' answers can and cannot tell us")
st.write("A dashboard for our MIS 545 group project. It uses the CDC Youth Risk Behavior Survey: models are built on the **2023** survey "
         "and checked once on the **2025** survey. The outcome is whether a student felt sad or hopeless almost every day for two weeks or more.")

s = students(); ml = table("model_ladder.csv"); st_ = table("staged_social_media.csv")
lr25 = ml.loc[ml.model == "Logistic regression", "test_auroc_2025"].iloc[0]
full = st_[(st_.outcome.str.startswith("Sad")) & (~st_.weighted) & (st_.stage.str.startswith("4"))].iloc[0]
c = st.columns(4)
c[0].metric("Students", f"{len(s):,}", f"{(s.year == 2023).sum():,} in 2023 · {(s.year == 2025).sum():,} in 2025", delta_color="off", delta_arrow="off")
c[1].metric("Felt sad or hopeless", f"{s.sad[s.year == 2023].mean():.0%} → {s.sad[s.year == 2025].mean():.0%}", "2023 → 2025, unweighted", delta_color="off", delta_arrow="off")
c[2].metric("Prediction score on 2025", f"{lr25:.2f}", "AUROC, logistic regression", delta_color="off", delta_arrow="off")
c[3].metric("Heaviest social media users", f"{full.odds:.2f}×", "odds of sadness once everything else is known", delta_color="off", delta_arrow="off")

st.divider()
st.subheader("What's inside")
pages = [
    ("pages/1_Social_Media.py", "📱 The social media story", "Watch the social media link shrink as we learn more about each student."),
    ("pages/2_Fairness.py", "⚖️ The fairness slider", "Move the risk cutoff and see who gets caught: girls versus boys."),
    ("pages/3_Student_Map.py", "🗺️ The student map", "Every student as a dot, colored by profile, sadness or social media use."),
    ("pages/4_What_Drives_Risk.py", "🎛️ What drives risk?", "Choose answers and see which ones push predicted risk up or down."),
    ("pages/5_Model_Scoreboard.py", "🏆 Model scoreboard", "Five models, the riskiest tenth, and the questions that matter most."),
    ("pages/6_Ask_the_Data.py", "💬 Ask the data", "Ask Claude questions; it answers only from our results."),
]
cols = st.columns(3)
for i, (path, label, text) in enumerate(pages):
    with cols[i % 3]:
        with st.container(border=True):
            st.page_link(path, label=label)
            st.caption(text)

st.subheader("The short version")
st.markdown("""
- **Prediction works.** A simple model rates 2025 students it never saw well (0.81). The riskiest tenth: 84% sad. The lowest tenth: 4%.
- **Ten questions do most of the work:** emotional abuse by a parent, sex, sexual identity, sleep, feeling close to people at school, bullying.
- **Social media is linked to sadness, but mostly through other things that go with heavy use.** Its link shrinks from 1.74× to 1.33× once the rest of a student's life is known.
- **One cutoff is not fair:** it catches 78% of sad girls but only 49% of sad boys. Separate cutoffs fix it.
""")
caveat()
