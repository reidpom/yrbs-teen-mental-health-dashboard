import hmac
import streamlit as st
from common import setup, caveat

setup("Ask the data", "💬")
st.title("💬 Ask the data")
st.write("Ask a question about the project. Claude answers only from our results tables and write-up, and shows which lookups it made.")


def secret(key, default=None):
    try:
        return st.secrets.get(key, default)
    except Exception:              # no secrets file at all
        return default


api_key = secret("ANTHROPIC_API_KEY")
model = secret("CLAUDE_MODEL", None)
max_q = int(secret("MAX_QUESTIONS_PER_VISITOR", 15))
password = secret("APP_PASSWORD")

if not api_key:
    st.info("The assistant is off: no API key is set. Add `ANTHROPIC_API_KEY` to `.streamlit/secrets.toml` (local) "
            "or to the app's Secrets on Streamlit Community Cloud. See the README.", icon="🔑")
    st.stop()

if password and not st.session_state.get("chat_unlocked"):
    pw = st.text_input("Password for the assistant", type="password", help="Ask a team member. This keeps the API bill under control.")
    if pw and hmac.compare_digest(pw, password):
        st.session_state.chat_unlocked = True
        st.rerun()
    elif pw:
        st.error("Wrong password.")
    st.stop()

import anthropic
import agent

MODEL = model or agent.DEFAULT_MODEL


@st.cache_resource
def client(key):
    return anthropic.Anthropic(api_key=key, max_retries=2, timeout=120)


ss = st.session_state
ss.setdefault("api_messages", []); ss.setdefault("display", []); ss.setdefault("asked", 0); ss.setdefault("spent", 0.0)

with st.sidebar:
    st.caption(f"Model: `{MODEL}` · effort: low")
    st.caption(f"Questions this visit: {ss.asked} of {max_q}")
    st.caption(f"Estimated cost this visit: ${ss.spent:.3f}")
    if st.button("Start a new conversation"):
        ss.api_messages, ss.display = [], []
        st.rerun()

examples = ["How much does social media matter compared with sleep and bullying?", "Is the model fair to boys?",
            "Which five questions predict sadness best, and how do we know?", "Did sadness go down from 2023 to 2025?"]
if not ss.display:
    st.markdown("**Try one of these:**")
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        if cols[i % 2].button(ex, width="stretch"):
            ss.pending = ex
            st.rerun()

for role, text, tools in ss.display:
    with st.chat_message(role):
        if tools:
            with st.expander(f"Looked up {len(tools)} thing{'s' if len(tools) > 1 else ''}"):
                st.code("\n".join(tools), language=None)
        st.markdown(text)

question = st.chat_input("Ask about the results…", disabled=ss.asked >= max_q) or ss.pop("pending", None)
if ss.asked >= max_q:
    st.warning(f"You've reached this visit's limit of {max_q} questions.")

if question:
    ss.asked += 1
    ss.display.append(("user", question, []))
    with st.chat_message("user"):
        st.markdown(question)
    ss.api_messages.append({"role": "user", "content": question})
    n_before = len(ss.api_messages)
    with st.chat_message("assistant"):
        status = st.status("Thinking…", expanded=False)
        box = st.empty(); text, tools, info = "", [], {}
        for kind, payload in agent.answer(client(api_key), MODEL, ss.api_messages):
            if kind == "text":
                text += payload; box.markdown(text + "▌")
            elif kind == "tool":
                tools.append(payload); status.update(label=f"Looking up: {payload}")
                text += "\n\n" if text and not text.endswith("\n") else ""
            else:
                info = payload
        status.update(label=f"Done · {len(tools)} lookup{'s' if len(tools) != 1 else ''}", state="complete")
        if info.get("error"):
            text = (text + "\n\n" if text else "") + f"⚠️ {info['error']}"
        box.markdown(text)
    ss.spent += info.get("cost", 0.0)
    if info.get("drop_turn") or info.get("truncated"):
        del ss.api_messages[n_before - 1:]           # keep the conversation valid: drop the unanswered turn entirely
    ss.display.append(("assistant", text, tools))
    st.rerun()

caveat("Answers come from the project's own results. The assistant can still make mistakes, so check key numbers against the other pages.")
