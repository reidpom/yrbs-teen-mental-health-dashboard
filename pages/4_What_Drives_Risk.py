import numpy as np
import plotly.graph_objects as go
import streamlit as st
from common import setup, caveat, whatif_model, style, RED, BLUE, MUTED

setup("What drives risk?", "🎛️")
st.title("🎛️ What drives risk?")
m = whatif_model()
st.write(f"Pick answers for a made-up student. A small model built on 2023 (the 10 questions that matter most, plus social media) "
         f"estimates the chance they felt sad or hopeless, and shows which answers pushed it up or down. "
         f"This small model scores {m['auroc_2025']:.2f} on 2025 students, close to the full model's 0.81.")
st.warning("**For illustration only.** This is not a screening or diagnostic tool, and it says nothing about any real student. "
           "If you or someone you know is struggling, call or text **988** (Suicide & Crisis Lifeline).", icon="⚠️")

if "whatif" not in st.session_state:
    st.session_state.whatif = {q["q"]: q["default"] for q in m["questions"]}
if st.button("Reset to the most common answers"):
    st.session_state.whatif = {q["q"]: q["default"] for q in m["questions"]}
    for q in m["questions"]:
        st.session_state.pop(f"wi_{q['q']}", None)

left, right = st.columns([2, 3])
picked = {}
with left:
    for q in m["questions"]:
        opts = [a["code"] for a in q["answers"]]; text = {a["code"]: a["text"] for a in q["answers"]}
        default = st.session_state.whatif.get(q["q"], q["default"])
        picked[q["q"]] = st.selectbox(q["name"], opts, index=opts.index(default) if default in opts else 0,
                                      format_func=lambda c, t=text: t[c], key=f"wi_{q['q']}", help=q["label"])
        st.session_state.whatif[q["q"]] = picked[q["q"]]

contrib = []
for q in m["questions"]:
    coef = {a["code"]: a["coef"] for a in q["answers"]}[picked[q["q"]]]
    contrib.append((q["name"], {a["code"]: a["text"] for a in q["answers"]}[picked[q["q"]]], coef - q["mean_contrib"]))
logit = m["base_logit"] + sum(c for _, _, c in contrib)
p, base_p = 1 / (1 + np.exp(-logit)), 1 / (1 + np.exp(-m["base_logit"]))

with right:
    a, b = st.columns(2)
    a.metric("Estimated chance of feeling sad or hopeless", f"{p:.0%}")
    b.metric("Average 2023 student (model baseline)", f"{base_p:.0%}", f"{(p - base_p) * 100:+.0f} points", delta_color="inverse")
    contrib.sort(key=lambda x: abs(x[2]))
    fig = go.Figure(go.Bar(y=[f"{n}: {t}" for n, t, _ in contrib], x=[c for _, _, c in contrib], orientation="h",
                           marker_color=[RED if c > 0 else BLUE for _, _, c in contrib],
                           hovertemplate="%{y}<br>push %{x:+.2f} (log-odds)<extra></extra>"))
    fig.add_vline(x=0, line_color=MUTED)
    fig.update_xaxes(title="← lowers predicted risk · raises it → (SHAP, log-odds)")
    fig.update_layout(title="Why the model rated this student this way")
    st.plotly_chart(style(fig, 520), width="stretch")
    st.caption("For a linear model these pushes are exact SHAP values: each answer's effect compared with an average 2023 student. "
               "Try changing social media use: its push is small next to emotional abuse, sleep or bullying.")
caveat("Pushes show what the model learned from 2023 answers. They describe links in the data, not what would happen if a student changed a habit.")
