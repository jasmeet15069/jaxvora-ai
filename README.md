# jaxvora-ai

Jaxvora AI deployment repository — an autonomous multi-agent AI platform (22 specialist agents across Engineering, Security, Data, Career, and Product divisions).

## Models

All agents default to **North Mini Code Free** via **OpenCode Zen** (`north-mini-code-free`). The Chief Orchestrator runs Groq-first for low latency, and Groq / OpenRouter (DeepSeek) remain in the failover chain. The default is set by `OPENCODE_ZEN_MODEL` (see `server/.env.example`). See [`JAXVORA_ARCHITECTURE.md`](JAXVORA_ARCHITECTURE.md) and [`AGENT_GRAPH.md`](AGENT_GRAPH.md) for diagrams.

## Layout

- `server/` - Python FastAPI application plus `server/api-server/` TypeScript API artifact.
- `frontend/` - Vite React frontend.
- `lib/` - Shared TypeScript API/database libraries.
- `scripts/` - Build and operational scripts.
- `main.py` - Backend entrypoint (mirror of `server/main.py`; the deployed service runs this).
