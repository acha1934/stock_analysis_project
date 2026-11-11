"""Model Layer (Part 1) — AI / Agent Core.

Modules
───────
tools.py      → Stock analysis functions + tool registry
retriever.py  → Semantic vector retrieval (sentence-transformers + FAISS)
agent.py      → LLM orchestrator (Groq API) for tool selection + synthesis
executor.py   → Sandboxed tool execution with error handling
"""
