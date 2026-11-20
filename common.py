"""Shared loaders, labels and styling for every page."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st

DATA = Path(__file__).resolve().parent / "data"
BLUE, ORANGE, NAVY, MUTED, RED, GRID = "#2A78D6", "#EB6834", "#14213D", "#6B6A63", "#E34948", "#E6E5DF"
GROUP_COLORS = ["#2A78D6", "#7FB2EC", "#F2B04B", "#EB6834", "#B8322B"]          # least to most sad

KMEANS_NAMES = {1: "1 · Low risk: not sexually active, little substance use", 2: "2 · Sexually active, little substance use",
                3: "3 · Vaping and drinking, more bullying", 4: "4 · Sexually active and substance use",
                5: "5 · Heavy multi-risk: daily vaping, marijuana, violence"}
RISK_NAMES = {1: "1 · Low risk: no vaping, drinking or marijuana", 2: "2 · Eats and drinks a lot of everything",
              3: "3 · Inactive, poor diet, no substance use", 4: "4 · Vaping, bullied, exposed to violence",
              5: "5 · Heavy multi-risk: daily vaping, heavy marijuana"}
WARD_NAMES = {k: f"{k} · Hierarchical group {k}" for k in range(1, 6)}
SOCIAL_ORDER = ["Does not use", "A few times a month", "About once a week", "A few times a week", "About once a day",
                "Several times a day", "About once an hour", "More than once an hour"]


@st.cache_data
def students() -> pd.DataFrame:
    return pd.read_csv(DATA / "students.csv")


@st.cache_data
def table(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / "tables" / name)


@st.cache_data
def question_stats() -> pd.DataFrame:
    return pd.read_csv(DATA / "question_stats.csv")


@st.cache_data
def questions() -> dict:
    return json.loads((DATA / "questions.json").read_text())


@st.cache_data
def test_risk() -> pd.DataFrame:
    return pd.read_csv(DATA / "test_risk_2025.csv")


@st.cache_data
def whatif_model() -> dict:
    return json.loads((DATA / "whatif_model.json").read_text())


def setup(title: str, icon: str = "📊"):
    st.set_page_config(page_title=f"{title} · Teen Mental Health", page_icon=icon, layout="wide")
    st.markdown("""<style>
      .block-container {padding-top: 2rem; max-width: 1200px;}
      [data-testid="stMetricValue"] {font-size: 2rem;}
      .caveat {font-size: .85rem; color: #6B6A63; border-left: 3px solid #E6E5DF; padding-left: .7rem; margin-top: 1.5rem;}
    </style>""", unsafe_allow_html=True)


def caveat(extra: str = ""):
    st.markdown(f'<div class="caveat">CDC Youth Risk Behavior Survey (national high school sample). Models are built on 2023 and tested once on 2025. '
                f'Each student answers once, so every result is a link, not a cause. {extra}</div>', unsafe_allow_html=True)


def style(fig, height=420):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="white", paper_bgcolor="white",
                      font=dict(family="Inter, Helvetica, Arial", color=NAVY), legend_title_text="")
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    return fig
