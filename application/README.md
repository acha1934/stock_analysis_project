# Application Layer — Stock Agent

Full-stack infrastructure wrapping the Model Layer (Part 1) with a
FastAPI backend, Streamlit frontend, and benchmarking harness.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy env config
cp .env.example .env

# 3. Start the FastAPI backend (terminal 1)
uvicorn main:app --reload --port 8000

# 4. Start the Streamlit frontend (terminal 2)
streamlit run app.py

# 5. Run benchmarks (terminal 3)
python benchmark.py --export results.json
```

## File Structure

```
application/
├── main.py              # FastAPI backend — REST endpoints
├── app.py               # Streamlit chat frontend
├── benchmark.py         # Evaluation & benchmarking harness
├── config.py            # Central settings (pydantic-settings)
├── schemas.py           # Pydantic request/response models
├── services/
│   └── agent_service.py # Bridge to Model Layer (Part 1)
├── static/
│   └── charts/          # Generated chart images
├── tests/
│   └── test_api.py       # API endpoint tests
├── requirements.txt
├── .env.example
└── README.md
```

## API Endpoints

| Method | Path             | Description                          |
|--------|------------------|--------------------------------------|
| POST   | `/api/chat`      | Accept query → return answer + chart |
| GET    | `/api/tools`     | List all registered tools            |
| POST   | `/api/benchmark` | Run evaluation over query suite     |
| GET    | `/api/health`    | Health check                         |

## Integration with Part 1

`services/agent_service.py` is the single bridge. When Part 1 modules
are ready, update the imports and method bodies to call:

- `tools.py` → `list_tools()` and tool definitions
- `retriever.py` → vector-based tool retrieval
- `agent.py` → LLM tool selection + parameter extraction
- `executor.py` → sandboxed code execution
