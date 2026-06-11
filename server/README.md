# Jaxvora

Jaxvora is a production-ready, autonomous multi-agent AI platform for Engineering, Data, Security, Career, and Automation workflows. It combines DeepSeek V3 (via OpenRouter) and Llama 3.3 70B (via Groq) in an orchestrated multi-agent system with a dark mission-control UI.

## Architecture Overview

```
User → Chat Interface → Chief Orchestrator (Llama 3.3 70B)
                              ↓
               Project Intelligence (Groq/Llama)
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
   - `OPENROUTER_API_KEY` — for DeepSeek V3 and other models
   - `GROQ_API_KEY` — Groq key for Llama 3.3 70B (orchestrator + code review)

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

| Division | Agent | Model |
|---|---|---|
| Executive | Chief Orchestrator | Llama 3.3 70B (Groq) |
| Executive | Project Intelligence | Llama 3.3 70B (Groq) |
| Engineering | AI Engineer | DeepSeek V3 |
| Engineering | Software Engineer | DeepSeek V3 |
| Engineering | Debug Agent | DeepSeek V3 |
| Engineering | QA/Test Agent | DeepSeek V3 |
| Engineering | Code Review | Llama 3.3 70B (Groq) |
| Engineering | Architecture | Llama 3.3 70B (Groq) |
| Engineering | Database | DeepSeek V3 |
| Engineering | DevOps | DeepSeek V3 |
| Security | Cybersecurity | DeepSeek V3 |
| Security | Red Team | DeepSeek V3 |
| Security | Compliance | DeepSeek V3 |
| Data | Data Analyst | DeepSeek V3 |
| Data | BI Agent | DeepSeek V3 |
| Data | Data Engineer | DeepSeek V3 |
| Data | ML Engineer | DeepSeek V3 |
| Career | Resume Agent | DeepSeek V3 |
| Career | Interview Coach | DeepSeek V3 |
| Career | Career Coach | DeepSeek V3 |
| Product | Product Manager | DeepSeek V3 |
| Product | Documentation | DeepSeek V3 |
| Product | Research | DeepSeek V3 |
