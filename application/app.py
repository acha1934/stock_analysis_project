"""Streamlit chat frontend — interactive web dashboard.

Run with:
    streamlit run app.py

Features
────────
• Clean chat interface with message history
• Renders text answers, formatted tables, and output charts
• Sidebar control panel to toggle routing mode (Naive vs. Vector RAG)
• Passes conversation history to the agent for context-aware follow-up queries
"""

from __future__ import annotations

import requests
import streamlit as st

# ── Page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Agent Chat",
    page_icon="📊",
    layout="wide",
)

API_BASE = "http://localhost:8000"


# ── Helper: build history payload from session messages ─────────────
def get_history() -> list[dict[str, str]]:
    """Return last 6 turns of chat history as [{role, content}] for the API."""
    return [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.get("messages", [])[-6:]
    ]


# ── Helper: call the chat API and render the response ───────────────
def call_api_and_render(prompt: str) -> None:
    """Send a query to the backend, render the response, and append to history."""
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/api/chat",
                    json={
                        "query": prompt,
                        "routing_mode": routing_mode,
                        "history": get_history(),
                    },
                    timeout=60,
                )
                if resp.ok:
                    data = resp.json()
                    st.markdown(data["answer"])
                    if data.get("chart_path"):
                        st.image(data["chart_path"])
                    st.caption(
                        f"Tool: `{data.get('tool_used', 'N/A')}` | "
                        f"Args: `{data.get('tool_args', {})}` | "
                        f"Latency: {data['latency_ms']:.0f}ms"
                    )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "tool": data.get("tool_used"),
                        "args": data.get("tool_args"),
                        "latency": data["latency_ms"],
                        "chart": data.get("chart_path"),
                    })
                else:
                    st.error(f"API error: {resp.status_code} — {resp.text}")
            except requests.RequestException as e:
                st.error(f"Could not reach API server: {e}")


# ── Sidebar: control panel ──────────────────────────────────────────
st.sidebar.title("⚙️ Control Panel")

routing_mode = st.sidebar.radio(
    "Routing Mode",
    options=["vector", "naive"],
    index=0,
    help="Vector RAG retrieves top-k tools via embeddings. Naive includes all tools in the prompt.",
)

st.sidebar.divider()

# Tool icons and example queries
TOOL_ICONS = {
    "get_stock_price": "💰",
    "compare_returns": "📊",
    "plot_stock_trend": "📈",
    "plot_comparison": "📉",
    "compare_multiple": "🔀",
}
TOOL_EXAMPLE_QUERIES = {
    "get_stock_price": "What is the latest price of TCS?",
    "compare_returns": "Compare the returns of TCS and Reliance for 2023",
    "plot_stock_trend": "Plot the stock trend of Infosys from 2023 to 2024",
    "plot_comparison": "Compare TCS and Infosys for 2023 in form of a graph",
    "compare_multiple": "Compare TCS, Infosys, Wipro and HCL Tech for the year 2023",
}

# Display available tools as buttons
st.sidebar.subheader("🛠️ Available Tools")
st.sidebar.caption("Click a tool to try an example query.")

if "prefill_query" not in st.session_state:
    st.session_state.prefill_query = ""

try:
    tools_resp = requests.get(f"{API_BASE}/api/tools", timeout=5)
    if tools_resp.ok:
        for tool in tools_resp.json():
            icon = TOOL_ICONS.get(tool["name"], "🔧")
            with st.sidebar.expander(f"{icon}  {tool['name']}", expanded=False):
                st.caption(tool["description"])
                if st.button(
                    "▶ Try example",
                    key=f"btn_{tool['name']}",
                    use_container_width=True,
                ):
                    st.session_state.prefill_query = TOOL_EXAMPLE_QUERIES.get(tool["name"], "")
                    st.rerun()
    else:
        st.sidebar.warning("Could not fetch tools. Is the API running?")
except requests.RequestException:
    st.sidebar.warning("API server not reachable. Start it with: `uvicorn main:app --reload`")


# ── Main chat interface ──────────────────────────────────────────────
st.title("📊 Stock Agent Chat")
st.caption("Ask questions about Indian stocks in natural language. Follow-up questions are supported.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("chart"):
            st.image(msg["chart"])
        if msg.get("tool"):
            st.caption(f"Tool: `{msg['tool']}` | Args: `{msg['args']}` | Latency: {msg['latency']:.0f}ms")

# Auto-submit when a sidebar tool button was clicked
if st.session_state.get("prefill_query"):
    prefill_prompt = st.session_state.pop("prefill_query")
    with st.chat_message("user"):
        st.markdown(prefill_prompt)
    st.session_state.messages.append({"role": "user", "content": prefill_prompt})
    call_api_and_render(prefill_prompt)

# Manual chat input
if prompt := st.chat_input("Ask about Indian stocks..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    call_api_and_render(prompt)
