# Jaxvora

Autonomous multi-agent AI platform for Engineering, Data, Security, Career, and Automation workflows. 22 specialist agents coordinated by Llama 3.3 70B (Groq), powered by DeepSeek V3 (OpenRouter).

## Run & Operate

- `cd jaxvora && /home/runner/workspace/.pythonlibs/bin/python3 -m uvicorn main:app --host 0.0.0.0 --port 8099` — run Jaxvora (served via Jaxvora workflow)
- `pnpm --filter @workspace/api-server run dev` — run the API server (port 5000)
- Required env secrets: `OPENROUTER_API_KEY`, `GROQ_API_KEY`, `DATABASE_URL`

## Stack

- **Backend**: Python 3.11 + FastAPI + asyncpg
- **Frontend**: Vanilla HTML/CSS/JS (dark command center UI)
- **Orchestrator**: Llama 3.3 70B via Groq
- **Worker agents**: DeepSeek V3 via OpenRouter
- **Review/Architecture**: Llama 3.3 70B via Groq
- **DB**: PostgreSQL (Neon) — tasks, logs, audit, knowledge_base (full-text search replaces Qdrant)
- **Real-time**: WebSockets (/ws/agents, /ws/tasks, /ws/logs, /ws/chat)
- Node.js workspace: pnpm workspaces, TypeScript 5.9, Express 5

## Where things live

- `jaxvora/main.py` — complete FastAPI backend (all 11 sections)
- `jaxvora/index.html` — dark mission-control frontend (served at `/`)
- `jaxvora/requirements.txt` — Python dependencies
- `lib/api-spec/openapi.yaml` — Node API contracts
- `artifacts/api-server/src/` — Express API server

## Architecture decisions

- Qdrant replaced with PostgreSQL full-text search (tsvector + pg_trgm) — no extra service needed
- No GitHub integration — removed per user request
- All agents share BaseAgent class; inter-agent calls route through Chief Orchestrator only
- OpenRouter used for all DeepSeek model calls (not api.deepseek.com directly)
- Graceful fallbacks for all missing API keys — never crashes

## Product

- Chat with Chief Orchestrator (Llama 3.3 70B) for any engineering/data/security/career task
- 22 agents across 5 divisions: Engineering, Security, Data, Career, Product
- Live agent monitor with real-time WebSocket status
- Task queue, live logs, approval center, analytics dashboard
- Persistent memory (PostgreSQL knowledge_base) with semantic-style search

## User preferences

- No Qdrant — use PostgreSQL full-text search instead
- No GitHub integration
- Use OpenRouter for DeepSeek models

## Gotchas

- Python binary is at `/home/runner/workspace/.pythonlibs/bin/python3`
- Jaxvora runs on port 8099 (8080 is in use by another service)
- DATABASE_URL must be set for persistent storage; app still runs without it but tasks/logs won't persist
