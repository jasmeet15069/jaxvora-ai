# Jaxvora

Jaxvora is a production-ready, autonomous multi-agent AI platform for Engineering, Data, Security, Career, and Automation workflows. Every agent's default brain is **North Mini Code (Free)** via **OpenCode Zen** (`north-mini-code-free`); the Chief Orchestrator runs Groq-first for low latency, and Groq / OpenRouter (DeepSeek) remain in the failover chain. Orchestrated multi-agent system with a dark mission-control UI.

## Architecture Overview

```
User → Chat Interface → Chief Orchestrator (Groq-first)
                              ↓
               Project Intelligence (North Mini Code Free)
                              ↓
    ┌─────────────────────────────────────────────┐
    │  Engineering  │  Security  │  Data  │ Career │
    │   Division    │  Division  │  Div.  │  Div.  │
    └─────────────────────────────────────────────┘
                              ↓
              PostgreSQL (Tasks, Logs, Memory, Audit)
```

All agents share a `BaseAgent` class. Inter-agent communication routes exclusively through the Chief Orchestrator. Every action is logged and auditable.

## Setup (Replit)

1. **Set secrets** in Replit Secrets panel:
   - `DATABASE_URL` — Neon PostgreSQL connection string
   - `OPENCODE_ZEN_API_KEY` — OpenCode Zen key; default brain for all agents (`north-mini-code-free`)
   - `GROQ_API_KEY` — Groq key for Llama 3.3 70B (orchestrator + failover)
   - `OPENROUTER_API_KEY` — DeepSeek failover

   Optional overrides: `OPENCODE_ZEN_MODEL` (default `north-mini-code-free`), `OPENCODE_ZEN_PRIMARY`, `LLM_PROVIDER_ORDER`, `ORCHESTRATOR_PROVIDER`.

2. **Run** via the workflow or:
   ```bash
   cd jaxvora && pip install -r requirements.txt
   uvicorn main:app --host 0.0.0.0 --port 8080 --reload
   ```

## Example Chat Commands

- `Fix all bugs in the authentication module`
- `Analyze the current architecture and suggest improvements`
- `Run a full security audit`
- `Generate an ATS-optimized resume for a senior backend engineer`
- `Optimize slow database queries`
- `Review code quality and identify technical debt`
- `Create a data pipeline for user analytics`
- `Write unit tests for the payment service`

## Agent Directory

All 22 agents default to **North Mini Code Free** via OpenCode Zen (`north-mini-code-free`). The Chief Orchestrator runs Groq-first for low latency; Groq and OpenRouter (DeepSeek) stay in the failover chain.

| Division | Agent | Default Model |
|---|---|---|
| Executive | Chief Orchestrator | North Mini Code Free (OpenCode Zen) · Groq-first |
| Executive | Project Intelligence | North Mini Code Free (OpenCode Zen) |
| Engineering | AI Engineer | North Mini Code Free (OpenCode Zen) |
| Engineering | Software Engineer | North Mini Code Free (OpenCode Zen) |
| Engineering | Debug Agent | North Mini Code Free (OpenCode Zen) |
| Engineering | QA/Test Agent | North Mini Code Free (OpenCode Zen) |
| Engineering | Code Review | North Mini Code Free (OpenCode Zen) |
| Engineering | Architecture | North Mini Code Free (OpenCode Zen) |
| Engineering | Database | North Mini Code Free (OpenCode Zen) |
| Engineering | DevOps | North Mini Code Free (OpenCode Zen) |
| Security | Cybersecurity | North Mini Code Free (OpenCode Zen) |
| Security | Red Team | North Mini Code Free (OpenCode Zen) |
| Security | Compliance | North Mini Code Free (OpenCode Zen) |
| Data | Data Analyst | North Mini Code Free (OpenCode Zen) |
| Data | BI Agent | North Mini Code Free (OpenCode Zen) |
| Data | Data Engineer | North Mini Code Free (OpenCode Zen) |
| Data | ML Engineer | North Mini Code Free (OpenCode Zen) |
| Career | Resume Agent | North Mini Code Free (OpenCode Zen) |
| Career | Interview Coach | North Mini Code Free (OpenCode Zen) |
| Career | Career Coach | North Mini Code Free (OpenCode Zen) |
| Product | Product Manager | North Mini Code Free (OpenCode Zen) |
| Product | Documentation | North Mini Code Free (OpenCode Zen) |
| Product | Research | North Mini Code Free (OpenCode Zen) |
