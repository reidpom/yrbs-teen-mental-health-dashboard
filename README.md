# Teen Mental Health dashboard (MIS 545)

A Streamlit dashboard for the YRBS project: train on 2023, test on 2025, outcome = felt sad or hopeless.

| Page | What it shows |
|---|---|
| Home | Headline numbers and the short version |
| 📱 Social media story | Raw sad rates by use, then the social media odds shrinking as sleep, bullying and everything else are added |
| ⚖️ Fairness slider | Move the risk cutoff (one cutoff or separate ones) and see how many sad girls vs boys are caught |
| 🗺️ Student map | UMAP map of ~25,000 students colored by k-means, hierarchical or risk-only groups, sadness or social media use |
| 🎛️ What drives risk? | Pick answers for a made-up student; see predicted risk and each answer's push (exact SHAP for a small linear model) |
| 🏆 Model scoreboard | Five models on 2023 vs 2025, boosting vs logistic regression, risk tenths, SHAP top questions, RFE curve |
| 💬 Ask the data | Claude answers questions using only the project's result tables and write-up (needs an API key) |

## Run it on your computer

```bash
pip install -r requirements.txt
streamlit run Home.py
```

The `data/` folder is already built. To rebuild it after rerunning the analysis: `python3 prepare_data.py ..` (needs the full project folder).

## Turn on the chat assistant

1. Get an API key at console.anthropic.com (set a monthly spend limit there).
2. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and paste the key. **Never commit this file** (it's in `.gitignore`).
3. Optional settings in the same file: `APP_PASSWORD` (only people with the password can use the chat), `MAX_QUESTIONS_PER_VISITOR` (default 15), `CLAUDE_MODEL` (default `claude-opus-5-5`; `claude-sonnet-5-5` or `claude-haiku-5-5` cost less).

How it works: Claude gets the project write-up (cached, so repeat questions cost less) and five read-only tools: list and read the result tables, search survey questions, break a question down by answer, and compute weighted sad rates by group. It cannot see individual students. It runs at low effort, with an 8,000-token answer limit and at most six lookups per question; the sidebar shows an estimated cost per visit. Refused requests fall back to another model automatically (`fallbacks: "default"`).

## Put it online (Streamlit Community Cloud, free)

1. Push this `dashboard/` folder to a GitHub repository (the data is public CDC data; `secrets.toml` stays out).
2. At share.streamlit.io: **Create app** → pick the repo → main file `Home.py`.
3. **Advanced settings → Secrets**: paste the contents of your `secrets.toml`.
4. Deploy. Share the link. Apps sleep after a few days without visits; opening the link wakes them in under a minute, so open it before presenting.
