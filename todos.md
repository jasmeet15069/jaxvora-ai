# Jaxvora AI — Roadmap

## Phase A — User System (HIGH)
- [ ] Login/auth (Google OAuth or email+password)
- [ ] Multi-user sessions with role-based access
- [ ] User-specific workspaces and chat history

## Phase B — Agent Memory & Persistence (HIGH)
- [ ] Agents remember past conversations and decisions
- [ ] Long-term memory via vector store (agent learns from experience)
- [ ] Session continuity across page refreshes

## Phase C — Scheduled & Recurring Tasks (HIGH)
- [ ] Cron-based agent scheduling (e.g. "run security audit every Monday")
- [ ] Background task queue with progress tracking
- [ ] Notification when scheduled tasks complete (email/push)

## Phase D — Agent Configuration UI (HIGH)
- [ ] Edit agent prompts from dashboard
- [ ] Swap models per agent
- [ ] Set temperature/max_tokens per agent
- [ ] Toggle which tools each agent can access
- [ ] Create custom agent personas without code

## Phase E — Monitoring & Observability (MEDIUM)
- [ ] Per-agent token/cost tracker (LLM spend dashboard)
- [ ] Latency heatmap — which agents are slowest
- [ ] Error rate trends over time
- [ ] Daily/weekly activity reports

## Phase F — Mobile-Responsive & Sharing (MEDIUM)
- [ ] Sidebar collapses on mobile, chat-first layout
- [ ] Share agent chat sessions via link
- [ ] Public agent playground (rate-limited)

## Phase G — Plugin System & Webhooks (LOW)
- [ ] External webhook triggers (GitHub push → trigger DevOps agent)
- [ ] Plugin API for custom tools
- [ ] Agent-to-agent webhook communication

## Integrations
### GitHub/GitLab (HIGH)
- [ ] PR review via agents
- [ ] Issue triage automation

### Slack/Discord (HIGH)
- [ ] Bot to chat with Jaxvora from messaging apps

### Jira/Linear (MEDIUM)
- [ ] Auto-create tickets from agent decisions

### Data (MEDIUM)
- [ ] Export/import chat history, workspace files, settings

## Agent Upgrades
- [ ] Swarm mode — multiple agents collaborate in parallel on one task (HIGH)
- [ ] Safe code execution sandbox — agents run Python/JS in isolated container (HIGH)
- [ ] Unified search across chat history + workspace files + knowledge base (MEDIUM)

## UX Improvements
- [ ] Onboarding wizard for new users (MEDIUM)
- [ ] File versioning in workspace — undo/redo changes (LOW)

## Infrastructure
- [ ] Public REST API for external apps to call agents (MEDIUM)
- [ ] Model fallback chains per agent — configurable in UI (MEDIUM)
- [ ] Agent benchmarking suite — test accuracy/speed (LOW)
