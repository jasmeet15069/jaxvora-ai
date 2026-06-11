# === SECTION 1: Imports & Config ===

import os
import json
import asyncio
import traceback
import uuid
import logging
import re
import subprocess
import tempfile
from datetime import datetime
from typing import Optional, List, Dict, Any, Set
from contextlib import asynccontextmanager
from pathlib import Path

import asyncpg
import httpx
import smtplib
import ssl as ssl_lib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jaxvora")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
DATABASE_URL = os.environ.get("DATABASE_URL", "")
PORT = int(os.environ.get("PORT", 8080))

GMAIL_SENDER = os.environ.get("GMAIL_SENDER", "")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")
NOTIFICATION_EMAIL = os.environ.get("NOTIFICATION_EMAIL", "")

SSH_HOST = os.environ.get("SSH_HOST", "")
SSH_PORT = int(os.environ.get("SSH_PORT", "22"))
SSH_USER = os.environ.get("SSH_USER", "")
SSH_PASSWORD = os.environ.get("SSH_PASSWORD", "")
SSH_KEY = os.environ.get("SSH_KEY", "")

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
DEEPSEEK_MODEL = "deepseek/deepseek-chat-v3-0324"
LLAMA_MODEL = "llama-3.3-70b-versatile"

# ── LLM helpers ────────────────────────────────────────────────────────────────

async def call_openrouter(system: str, user: str, model: str = DEEPSEEK_MODEL) -> str:
    if not OPENROUTER_API_KEY:
        return f"[Mock response — OPENROUTER_API_KEY not set] Task: {user[:100]}"
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"{OPENROUTER_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "HTTP-Referer": "https://jaxvora.ai",
                    "X-Title": "Jaxvora",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": 2048,
                },
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[OpenRouter error: {e}]"


async def call_groq(system: str, user: str) -> str:
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set — falling back to OpenRouter")
        return await call_openrouter(system, user)
    delays = [1, 3, 7]
    for attempt, delay in enumerate(delays + [None]):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json={
                        "model": LLAMA_MODEL,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "max_tokens": 2048,
                    },
                )
                if r.status_code == 429:
                    if delay is not None:
                        logger.warning(f"Groq rate-limited (429), retrying in {delay}s (attempt {attempt+1}/3)")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.warning("Groq rate-limited after 3 retries — falling back to OpenRouter")
                        return await call_openrouter(system, user)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError:
            raise
        except Exception as e:
            if delay is not None:
                await asyncio.sleep(delay)
                continue
            logger.error(f"Groq error after retries: {e} — falling back to OpenRouter")
            return await call_openrouter(system, user)
    return await call_openrouter(system, user)


async def send_gmail(to_email: str, subject: str, body: str, attachment_name: str = "", attachment_data: bytes = b"") -> str:
    """Send email via Gmail SMTP app password."""
    if not GMAIL_SENDER or not GMAIL_APP_PASSWORD:
        return "[Gmail not configured — set GMAIL_SENDER and GMAIL_APP_PASSWORD]"
    if not to_email:
        return "[No recipient email — set NOTIFICATION_EMAIL in Settings]"
    try:
        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = f"Jaxvora <{GMAIL_SENDER}>"
        msg["To"] = to_email
        msg.attach(MIMEText(body, "plain"))
        if attachment_data and attachment_name:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment_data)
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f'attachment; filename="{attachment_name}"')
            msg.attach(part)
        ctx = ssl_lib.create_default_context()
        def _send():
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as srv:
                srv.login(GMAIL_SENDER, GMAIL_APP_PASSWORD)
                srv.sendmail(GMAIL_SENDER, to_email, msg.as_string())
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _send)
        logger.info(f"✉ Email sent to {to_email}: {subject}")
        return f"Email sent to {to_email}"
    except Exception as e:
        logger.error(f"Gmail error: {e}")
        return f"[Gmail error: {e}]"


# === SECTION 2: Database Layer ================================================

db_pool: Optional[asyncpg.Pool] = None

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    repo_url TEXT DEFAULT '',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id) ON DELETE SET NULL,
    agent_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    input TEXT NOT NULL DEFAULT '',
    output TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID REFERENCES tasks(id) ON DELETE CASCADE,
    level TEXT NOT NULL DEFAULT 'INFO',
    message TEXT NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS audit (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_name TEXT NOT NULL,
    action TEXT NOT NULL,
    payload TEXT DEFAULT '',
    approved BOOLEAN DEFAULT NULL,
    decision_reason TEXT DEFAULT '',
    timestamp TIMESTAMPTZ DEFAULT NOW()
);
ALTER TABLE audit ADD COLUMN IF NOT EXISTS decision_reason TEXT DEFAULT '';

CREATE TABLE IF NOT EXISTS agent_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_name TEXT NOT NULL,
    task_summary TEXT NOT NULL,
    outcome TEXT NOT NULL DEFAULT 'success',
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS knowledge_base (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    collection TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    search_vector TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS knowledge_base_fts ON knowledge_base USING GIN(search_vector);
CREATE INDEX IF NOT EXISTS knowledge_base_trgm ON knowledge_base USING GIN(content gin_trgm_ops);
"""

async def get_db() -> asyncpg.Pool:
    global db_pool
    if db_pool is None:
        raise RuntimeError("Database pool not initialised")
    return db_pool


async def db_execute(query: str, *args):
    pool = await get_db()
    async with pool.acquire() as conn:
        return await conn.execute(query, *args)


async def db_fetch(query: str, *args) -> List[asyncpg.Record]:
    pool = await get_db()
    async with pool.acquire() as conn:
        return await conn.fetch(query, *args)


async def db_fetchrow(query: str, *args) -> Optional[asyncpg.Record]:
    pool = await get_db()
    async with pool.acquire() as conn:
        return await conn.fetchrow(query, *args)


async def log_to_db(level: str, message: str, task_id: Optional[str] = None):
    try:
        await db_execute(
            "INSERT INTO logs (task_id, level, message) VALUES ($1, $2, $3)",
            task_id, level, message
        )
    except Exception:
        pass
    await ws_manager.broadcast_log(level, message)


# === SECTION 3: MCP Tool Registry =============================================

class MCPTool:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    async def run(self, params: Dict[str, Any]) -> str:
        raise NotImplementedError


class FileSystemTool(MCPTool):
    def __init__(self):
        super().__init__("file_system", "Read and write files in the workspace")

    async def run(self, params: Dict[str, Any]) -> str:
        try:
            action = params.get("action", "read")
            path = params.get("path", "")
            if action == "read":
                with open(path) as f:
                    return f.read()
            elif action == "write":
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w") as f:
                    f.write(params.get("content", ""))
                return f"Written to {path}"
            return "Unknown action"
        except Exception as e:
            return f"file_system error: {e}"


class TerminalTool(MCPTool):
    ALLOWED = re.compile(r'^(ls|cat|echo|pwd|find|grep|wc|head|tail|python3? -c|pip show)\b')

    def __init__(self):
        super().__init__("terminal", "Run sandboxed shell commands (read-only)")

    async def run(self, params: Dict[str, Any]) -> str:
        cmd = params.get("command", "")
        if not self.ALLOWED.match(cmd):
            return f"Command blocked for safety: {cmd}"
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=10
            )
            return result.stdout or result.stderr or "(no output)"
        except subprocess.TimeoutExpired:
            return "Command timed out"
        except Exception as e:
            return f"terminal error: {e}"


class PostgreSQLTool(MCPTool):
    def __init__(self):
        super().__init__("postgresql", "Execute SQL queries against the database")

    async def run(self, params: Dict[str, Any]) -> str:
        sql = params.get("query", "")
        if not sql.strip().lower().startswith("select"):
            return "Only SELECT queries allowed via MCP tool"
        try:
            rows = await db_fetch(sql)
            return json.dumps([dict(r) for r in rows], default=str)
        except Exception as e:
            return f"postgresql error: {e}"


class BrowserTool(MCPTool):
    def __init__(self):
        super().__init__("browser", "Fetch and parse web pages")

    async def run(self, params: Dict[str, Any]) -> str:
        url = params.get("url", "")
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                r = await client.get(url, headers={"User-Agent": "Jaxvora/1.0"})
                text = r.text[:3000]
                text = re.sub(r'<[^>]+>', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                return text
        except Exception as e:
            return f"browser error: {e}"


class SecurityScannerTool(MCPTool):
    PATTERNS = [
        (r'(?i)(password|passwd|pwd)\s*=\s*["\'][^"\']{3,}', "Hardcoded password"),
        (r'(?i)(api_?key|apikey|secret)\s*=\s*["\'][^"\']{8,}', "Hardcoded API key"),
        (r'(?i)(token)\s*=\s*["\'][^"\']{8,}', "Hardcoded token"),
        (r'-----BEGIN (RSA |EC )?PRIVATE KEY-----', "Private key in source"),
        (r'(?i)eval\s*\(', "eval() usage"),
        (r'(?i)exec\s*\(', "exec() usage"),
        (r'(?i)subprocess\.call\([^)]+shell\s*=\s*True', "shell=True in subprocess"),
    ]

    def __init__(self):
        super().__init__("security_scanner", "Static analysis for secrets and vulnerabilities")

    async def run(self, params: Dict[str, Any]) -> str:
        content = params.get("content", "")
        findings = []
        for pattern, desc in self.PATTERNS:
            if re.search(pattern, content):
                findings.append(f"⚠ {desc}")
        return "\n".join(findings) if findings else "✓ No obvious issues found"


class CodeFormatterTool(MCPTool):
    def __init__(self):
        super().__init__("code_formatter", "Lint and format code suggestions")

    async def run(self, params: Dict[str, Any]) -> str:
        lang = params.get("language", "python")
        code = params.get("code", "")
        lines = code.split("\n")
        issues = []
        for i, line in enumerate(lines, 1):
            if len(line) > 120:
                issues.append(f"Line {i}: exceeds 120 chars ({len(line)})")
            if "\t" in line and lang == "python":
                issues.append(f"Line {i}: tab indentation (use spaces)")
        return "\n".join(issues) if issues else "✓ Code looks clean"


class EmailNotificationTool(MCPTool):
    def __init__(self):
        super().__init__("email_notify", "Send email notifications for bugs, issues, and alerts to the configured recipient")

    async def run(self, params: Dict[str, Any]) -> str:
        to = params.get("to", NOTIFICATION_EMAIL)
        subject = params.get("subject", "Jaxvora Alert")
        body = params.get("body", "")
        return await send_gmail(to, subject, body)


class SSHTool(MCPTool):
    def __init__(self):
        super().__init__("ssh_exec", "Execute commands on a remote server via SSH for 24/7 monitoring and management")

    async def run(self, params: Dict[str, Any]) -> str:
        host = params.get("host", SSH_HOST)
        port = int(params.get("port", SSH_PORT))
        user = params.get("user", SSH_USER)
        password = params.get("password", SSH_PASSWORD)
        command = params.get("command", "")
        if not host or not user or not command:
            return "[SSH not configured — provide host, user, and command]"
        try:
            import asyncssh
            conn_kwargs: Dict[str, Any] = {
                "host": host, "port": port, "username": user,
                "known_hosts": None,
            }
            if password:
                conn_kwargs["password"] = password
            elif SSH_KEY:
                import io
                conn_kwargs["client_keys"] = [asyncssh.import_private_key(SSH_KEY)]
            async with asyncssh.connect(**conn_kwargs) as conn:
                result = await conn.run(command, timeout=30)
                output = result.stdout or result.stderr or "(no output)"
                return output[:3000]
        except ImportError:
            return "[asyncssh not installed]"
        except Exception as e:
            return f"[SSH error: {e}]"


class MCPToolRegistry:
    def __init__(self):
        self._tools: Dict[str, MCPTool] = {}

    def register(self, tool: MCPTool):
        self._tools[tool.name] = tool

    async def run(self, name: str, params: Dict[str, Any]) -> str:
        if name not in self._tools:
            return f"Tool '{name}' not found"
        return await self._tools[name].run(params)

    def list_tools(self) -> List[Dict]:
        return [{"name": t.name, "description": t.description} for t in self._tools.values()]


tool_registry = MCPToolRegistry()


# === SECTION 4: Memory Manager ================================================

class MemoryManager:
    COLLECTIONS = ["architecture_knowledge", "security_findings", "code_fixes", "org_knowledge"]

    async def store(self, collection: str, content: str, metadata: Dict = None):
        try:
            await db_execute(
                "INSERT INTO knowledge_base (collection, content, metadata) VALUES ($1, $2, $3)",
                collection, content, json.dumps(metadata or {})
            )
        except Exception as e:
            logger.warning(f"Memory store error: {e}")

    async def search(self, query: str, collection: Optional[str] = None, limit: int = 5) -> List[Dict]:
        try:
            if collection:
                rows = await db_fetch(
                    """SELECT id, collection, content, metadata, created_at,
                              ts_rank(search_vector, plainto_tsquery('english', $1)) AS rank
                       FROM knowledge_base
                       WHERE collection = $2
                         AND search_vector @@ plainto_tsquery('english', $1)
                       ORDER BY rank DESC LIMIT $3""",
                    query, collection, limit
                )
            else:
                rows = await db_fetch(
                    """SELECT id, collection, content, metadata, created_at,
                              ts_rank(search_vector, plainto_tsquery('english', $1)) AS rank
                       FROM knowledge_base
                       WHERE search_vector @@ plainto_tsquery('english', $1)
                       ORDER BY rank DESC LIMIT $2""",
                    query, limit
                )
            return [
                {"id": str(r["id"]), "collection": r["collection"],
                 "content": r["content"], "score": float(r["rank"])}
                for r in rows
            ]
        except Exception as e:
            logger.warning(f"Memory search error: {e}")
            return []


memory = MemoryManager()


# === SECTION 5: Agent Registry & Base Agent Class =============================

class AgentResult:
    def __init__(self, agent: str, task: str, output: str, success: bool = True, task_id: str = None):
        self.agent = agent
        self.task = task
        self.output = output
        self.success = success
        self.task_id = task_id or str(uuid.uuid4())

    def to_dict(self):
        return {
            "agent": self.agent, "task": self.task,
            "output": self.output, "success": self.success,
            "task_id": self.task_id
        }


class BaseAgent:
    name: str = "base"
    model: str = "deepseek"
    division: str = "general"
    description: str = ""
    _status: str = "idle"
    _current_task: str = ""

    async def call_llm(self, system: str, user: str) -> str:
        if self.model == "groq":
            return await call_groq(system, user)
        else:
            return await call_openrouter(system, user)

    async def run(self, task: str, project_id: Optional[str] = None) -> AgentResult:
        self._status = "running"
        self._current_task = task
        task_id = str(uuid.uuid4())
        try:
            await db_execute(
                "INSERT INTO tasks (id, project_id, agent_name, status, input) VALUES ($1, $2, $3, 'running', $4)",
                task_id, project_id, self.name, task
            )
            await log_to_db("INFO", f"[{self.name}] Starting: {task[:80]}", task_id)
            await ws_manager.broadcast_agent_status(self.name, "running", task[:60])
            output = await self._execute(task)
            await db_execute(
                "UPDATE tasks SET status='completed', output=$1, completed_at=NOW() WHERE id=$2",
                output, task_id
            )
            await db_execute(
                "INSERT INTO agent_history (agent_name, task_summary, outcome) VALUES ($1, $2, 'success')",
                self.name, task[:120]
            )
            await log_to_db("INFO", f"[{self.name}] Completed ✓", task_id)
            self._status = "idle"
            self._current_task = ""
            await ws_manager.broadcast_agent_status(self.name, "idle", "")
            return AgentResult(self.name, task, output, True, task_id)
        except Exception as e:
            tb = traceback.format_exc()
            await db_execute(
                "UPDATE tasks SET status='failed', output=$1, completed_at=NOW() WHERE id=$2",
                str(e), task_id
            )
            await log_to_db("ERROR", f"[{self.name}] Error: {e}", task_id)
            self._status = "error"
            await ws_manager.broadcast_agent_status(self.name, "error", str(e)[:60])
            return AgentResult(self.name, task, f"Error: {e}\n{tb}", False, task_id)

    async def _execute(self, task: str) -> str:
        raise NotImplementedError

    def status_dict(self):
        return {
            "name": self.name, "model": self.model,
            "division": self.division, "description": self.description,
            "status": self._status, "current_task": self._current_task
        }


# === SECTION 6: Agent Implementations =========================================

class AIEngineerAgent(BaseAgent):
    name = "AI Engineer"; model = "deepseek"; division = "Engineering"
    description = "AI features, RAG systems, MCP integrations, LLM workflows"
    async def _execute(self, task):
        return await self.call_llm(
            "You are an expert AI engineer specialising in LLM integrations, RAG systems, "
            "embeddings, vector databases, MCP tool design, and production AI workflows. "
            "Provide detailed, actionable technical guidance.",
            task
        )

class SoftwareEngineerAgent(BaseAgent):
    name = "Software Engineer"; model = "deepseek"; division = "Engineering"
    description = "Backend/frontend dev, CRUD, API generation"
    async def _execute(self, task):
        return await self.call_llm(
            "You are a senior full-stack software engineer. You write clean, production-ready "
            "code with best practices, proper error handling, and comprehensive comments.",
            task
        )

class DebugAgent(BaseAgent):
    name = "Debug Agent"; model = "deepseek"; division = "Engineering"
    description = "Root-cause analysis, log investigation, automated bug fixing"
    async def _execute(self, task):
        return await self.call_llm(
            "You are an expert debugger. You perform systematic root-cause analysis, read "
            "stack traces, investigate logs, and provide precise bug fixes with explanations.",
            task
        )

class QATestAgent(BaseAgent):
    name = "QA/Test Agent"; model = "deepseek"; division = "Engineering"
    description = "Unit, integration, E2E, regression tests"
    async def _execute(self, task):
        return await self.call_llm(
            "You are a QA automation engineer. You write comprehensive test suites covering "
            "unit, integration, and E2E scenarios. Use pytest, Jest, or appropriate frameworks.",
            task
        )

class CodeReviewAgent(BaseAgent):
    name = "Code Review"; model = "groq"; division = "Engineering"
    description = "Code quality, best practices, risk & security review"
    async def _execute(self, task):
        return await self.call_llm(
            "You are a senior code reviewer. You evaluate code for quality, security, "
            "performance, maintainability, and adherence to best practices. Be specific.",
            task
        )

class ArchitectureAgent(BaseAgent):
    name = "Architecture"; model = "groq"; division = "Engineering"
    description = "System design, scalability, technical debt"
    async def _execute(self, task):
        return await self.call_llm(
            "You are a principal systems architect. You design scalable, resilient systems "
            "and identify technical debt, single points of failure, and improvement areas.",
            task
        )

class DatabaseAgent(BaseAgent):
    name = "Database"; model = "deepseek"; division = "Engineering"
    description = "Query optimisation, schema design, migrations"
    async def _execute(self, task):
        return await self.call_llm(
            "You are a database expert specialising in PostgreSQL, query optimisation, "
            "schema design, indexing strategies, and zero-downtime migrations.",
            task
        )

class DevOpsAgent(BaseAgent):
    name = "DevOps"; model = "deepseek"; division = "Engineering"
    description = "CI/CD, Docker configs, Kubernetes manifests, deployments"
    async def _execute(self, task):
        return await self.call_llm(
            "You are a DevOps/SRE engineer. You design CI/CD pipelines, write Dockerfiles, "
            "Kubernetes manifests, Terraform configs, and automate deployments reliably.",
            task
        )

class CybersecurityAgent(BaseAgent):
    name = "Cybersecurity"; model = "deepseek"; division = "Security"
    description = "Vulnerability scanning, secret detection, hardening"
    async def _execute(self, task):
        result = await self.call_llm(
            "You are a cybersecurity engineer. You identify vulnerabilities, detect exposed "
            "secrets, recommend hardening measures, and produce actionable security reports.",
            task
        )
        await memory.store("security_findings", f"Task: {task}\nFindings: {result[:500]}")
        return result

class RedTeamAgent(BaseAgent):
    name = "Red Team"; model = "deepseek"; division = "Security"
    description = "Threat modelling, attack simulation"
    async def _execute(self, task):
        return await self.call_llm(
            "You are a red team security expert. You perform threat modelling, identify "
            "attack vectors, and simulate adversarial scenarios to strengthen defences.",
            task
        )

class ComplianceAgent(BaseAgent):
    name = "Compliance"; model = "deepseek"; division = "Security"
    description = "GDPR, SOC2, ISO27001 checklists"
    async def _execute(self, task):
        return await self.call_llm(
            "You are a compliance officer specialising in GDPR, SOC2, ISO27001, and HIPAA. "
            "Produce detailed compliance checklists and gap analysis reports.",
            task
        )

class DataAnalystAgent(BaseAgent):
    name = "Data Analyst"; model = "deepseek"; division = "Data"
    description = "SQL analysis, KPI tracking, business insights"
    async def _execute(self, task):
        return await self.call_llm(
            "You are a senior data analyst. You write SQL queries, build KPI dashboards, "
            "interpret trends, and translate data into clear business insights.",
            task
        )

class BIAgent(BaseAgent):
    name = "BI Agent"; model = "deepseek"; division = "Data"
    description = "Power BI reports, DAX generation, semantic model analysis"
    async def _execute(self, task):
        return await self.call_llm(
            "You are a Business Intelligence expert specialising in Power BI, Tableau, "
            "DAX formulas, semantic models, and executive dashboard design.",
            task
        )

class DataEngineerAgent(BaseAgent):
    name = "Data Engineer"; model = "deepseek"; division = "Data"
    description = "ETL pipelines, data quality, warehouse optimisation"
    async def _execute(self, task):
        return await self.call_llm(
            "You are a data engineer. You design ETL/ELT pipelines with dbt, Spark, or "
            "Airflow, enforce data quality contracts, and optimise warehouse performance.",
            task
        )

class MLEngineerAgent(BaseAgent):
    name = "ML Engineer"; model = "deepseek"; division = "Data"
    description = "Feature engineering, model training, evaluation pipelines"
    async def _execute(self, task):
        return await self.call_llm(
            "You are an ML engineer. You design feature pipelines, train and evaluate models, "
            "handle model versioning, and deploy ML systems to production.",
            task
        )

class ResumeAgent(BaseAgent):
    name = "Resume Agent"; model = "deepseek"; division = "Career"
    description = "ATS-optimised resume and portfolio generation"
    async def _execute(self, task):
        return await self.call_llm(
            "You are a professional resume writer and career coach. You create ATS-optimised "
            "resumes and portfolios that highlight achievements with quantified impact.",
            task
        )

class InterviewCoachAgent(BaseAgent):
    name = "Interview Coach"; model = "deepseek"; division = "Career"
    description = "Technical, behavioural, and mock interview prep"
    async def _execute(self, task):
        return await self.call_llm(
            "You are an expert interview coach for tech roles. You prepare candidates for "
            "system design, coding, and behavioural interviews with detailed coaching.",
            task
        )

class CareerCoachAgent(BaseAgent):
    name = "Career Coach"; model = "deepseek"; division = "Career"
    description = "Learning plans, skill-gap analysis, career guidance"
    async def _execute(self, task):
        return await self.call_llm(
            "You are a senior tech career coach. You analyse skill gaps, design 90-day "
            "learning plans, and provide strategic career progression guidance.",
            task
        )

class ProductManagerAgent(BaseAgent):
    name = "Product Manager"; model = "deepseek"; division = "Product"
    description = "Roadmaps, feature planning, user stories"
    async def _execute(self, task):
        return await self.call_llm(
            "You are a senior product manager. You create roadmaps, write user stories with "
            "acceptance criteria, prioritise backlogs, and align stakeholders.",
            task
        )

class DocumentationAgent(BaseAgent):
    name = "Documentation"; model = "deepseek"; division = "Product"
    description = "Technical docs, API docs, architecture docs"
    async def _execute(self, task):
        return await self.call_llm(
            "You are a technical writer. You produce clear, comprehensive documentation "
            "including API references, architecture docs, and developer guides.",
            task
        )

class ResearchAgent(BaseAgent):
    name = "Research"; model = "deepseek"; division = "Product"
    description = "Technology research, framework comparison"
    async def _execute(self, task):
        return await self.call_llm(
            "You are a technology researcher. You compare frameworks, evaluate libraries, "
            "assess trade-offs, and produce well-structured research reports.",
            task
        )

class ProjectIntelligenceAgent(BaseAgent):
    name = "Project Intelligence"; model = "groq"; division = "Executive"
    description = "Dependency graph, architecture graph, impact analysis"
    async def _execute(self, task):
        result = await self.call_llm(
            "You are a project intelligence system. You analyse codebases, build dependency "
            "graphs, assess change impact, and provide architectural context.",
            task
        )
        await memory.store("architecture_knowledge", f"Analysis: {result[:600]}")
        return result


async def auto_decide_audit(audit_id: str, agent_name: str, action: str, payload: str):
    """LLM reviews every audit entry and auto-approves or rejects — no human needed."""
    system = """You are Jaxvora's autonomous decision engine. You review agent actions and decide whether to APPROVE or REJECT them.

Rules:
- APPROVE: legitimate engineering, data, security, career, or research tasks.
- REJECT: requests that are harmful, illegal, unethical, or clearly out of scope.
- Almost everything a user asks an AI platform to do should be APPROVED.

Respond with ONLY valid JSON: {"decision": "APPROVE", "reason": "one-sentence explanation"}
or {"decision": "REJECT", "reason": "one-sentence explanation"}"""
    prompt = f"Agent: {agent_name}\nAction: {action}\nPayload: {payload[:400]}"
    try:
        raw = await call_groq(system, prompt)
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            approved = data.get("decision", "APPROVE").upper() == "APPROVE"
            reason = data.get("reason", "Auto-processed by LLM.")
        else:
            approved = True
            reason = "Auto-approved (LLM response unparseable)."
    except Exception as e:
        approved = True
        reason = f"Auto-approved (error: {str(e)[:80]})."
    try:
        await db_execute(
            "UPDATE audit SET approved=$1, decision_reason=$2 WHERE id=$3",
            approved, reason, audit_id
        )
        logger.info(f"Audit {audit_id}: {'APPROVED' if approved else 'REJECTED'} — {reason}")
    except Exception as e:
        logger.warning(f"Failed to update audit decision: {e}")


AGENT_REGISTRY: Dict[str, BaseAgent] = {}

def build_registry():
    agents = [
        AIEngineerAgent(), SoftwareEngineerAgent(), DebugAgent(), QATestAgent(),
        CodeReviewAgent(), ArchitectureAgent(), DatabaseAgent(), DevOpsAgent(),
        CybersecurityAgent(), RedTeamAgent(), ComplianceAgent(),
        DataAnalystAgent(), BIAgent(), DataEngineerAgent(), MLEngineerAgent(),
        ResumeAgent(), InterviewCoachAgent(), CareerCoachAgent(),
        ProductManagerAgent(), DocumentationAgent(), ResearchAgent(),
        ProjectIntelligenceAgent(),
    ]
    for a in agents:
        AGENT_REGISTRY[a.name] = a


# === SECTION 7: Chief Orchestrator ============================================

class ChiefOrchestrator:
    name = "Chief Orchestrator"
    model = "groq"

    SYSTEM = """You are Jaxvora's Chief Orchestrator powered by Llama 3.3 70B.
Your role: parse user intent, create execution plans, route to specialist agents, and synthesise results.

Available agents:
Engineering: AI Engineer, Software Engineer, Debug Agent, QA/Test Agent, Code Review, Architecture, Database, DevOps
Security: Cybersecurity, Red Team, Compliance
Data: Data Analyst, BI Agent, Data Engineer, ML Engineer
Career: Resume Agent, Interview Coach, Career Coach
Product: Product Manager, Documentation, Research
Executive: Project Intelligence

Always respond in this JSON format:
{
  "plan": "1-sentence execution plan",
  "agents": ["Agent Name 1", "Agent Name 2"],
  "response": "Your synthesised response to the user"
}"""

    async def process(self, user_input: str, stream_fn=None) -> Dict:
        try:
            raw = await call_groq(self.SYSTEM, user_input)
            # Extract JSON
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                plan = json.loads(match.group())
            else:
                plan = {"plan": "Direct response", "agents": [], "response": raw}
        except Exception as e:
            plan = {
                "plan": "Fallback direct response",
                "agents": [],
                "response": f"I'll help you with that. {user_input[:100]}... (Orchestrator note: {e})"
            }

        # Run assigned agents
        results = []
        for agent_name in plan.get("agents", [])[:3]:
            agent = AGENT_REGISTRY.get(agent_name)
            if agent:
                if stream_fn:
                    await stream_fn({"type": "agent_start", "agent": agent_name})
                result = await agent.run(user_input)
                results.append(result.to_dict())
                if stream_fn:
                    await stream_fn({"type": "agent_done", "agent": agent_name, "output": result.output[:200]})

        # Log to audit and auto-decide
        try:
            row = await db_fetchrow(
                "INSERT INTO audit (agent_name, action, payload) VALUES ($1, $2, $3) RETURNING id",
                "Chief Orchestrator", "process_request", user_input[:500]
            )
            if row:
                asyncio.create_task(auto_decide_audit(str(row["id"]), "Chief Orchestrator", "process_request", user_input[:500]))
        except Exception:
            pass

        final = plan.get("response", "Task completed.")
        if results:
            final += "\n\n---\n**Agent Results:**\n"
            for r in results:
                final += f"\n**{r['agent']}**: {r['output'][:300]}...\n"

        return {"plan": plan.get("plan", ""), "agents": plan.get("agents", []), "response": final, "results": results}


orchestrator = ChiefOrchestrator()


# === SECTION 8: WebSocket Manager =============================================

class WebSocketManager:
    def __init__(self):
        self.agents: Set[WebSocket] = set()
        self.tasks_ws: Set[WebSocket] = set()
        self.logs_ws: Set[WebSocket] = set()
        self.chat_ws: Set[WebSocket] = set()

    async def _send(self, connections: Set[WebSocket], data: Dict):
        dead = set()
        for ws in connections:
            try:
                await ws.send_text(json.dumps(data, default=str))
            except Exception:
                dead.add(ws)
        connections -= dead

    async def broadcast_agent_status(self, name: str, status: str, task: str):
        await self._send(self.agents, {"type": "agent_status", "name": name, "status": status, "task": task, "ts": datetime.now().isoformat()})

    async def broadcast_task(self, task: Dict):
        await self._send(self.tasks_ws, {"type": "task_update", **task})

    async def broadcast_log(self, level: str, message: str):
        await self._send(self.logs_ws, {"type": "log", "level": level, "message": message, "ts": datetime.now().isoformat()})

    async def send_chat(self, ws: WebSocket, data: Dict):
        try:
            await ws.send_text(json.dumps(data, default=str))
        except Exception:
            pass


ws_manager = WebSocketManager()


# === SECTION 9: FastAPI App + REST Endpoints ==================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await startup()
    yield
    # Shutdown
    global db_pool
    if db_pool:
        await db_pool.close()


app = FastAPI(title="Jaxvora", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Pydantic models ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    project_id: Optional[str] = None

class ProjectCreate(BaseModel):
    name: str
    repo_url: Optional[str] = ""
    metadata: Optional[Dict] = {}

class MemorySearch(BaseModel):
    query: str
    collection: Optional[str] = None
    limit: Optional[int] = 5

class SecurityScanRequest(BaseModel):
    content: str

class NotificationEmailRequest(BaseModel):
    email: str

class SSHConfigRequest(BaseModel):
    host: str
    port: Optional[int] = 22
    user: str
    password: Optional[str] = ""
    key: Optional[str] = ""

class SendEmailRequest(BaseModel):
    to: str
    subject: str
    body: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def frontend():
    with open(Path(__file__).parent / "index.html") as f:
        return f.read()


@app.post("/chat")
async def chat(req: ChatRequest):
    result = await orchestrator.process(req.message)
    return result


@app.get("/agents")
async def list_agents():
    return [a.status_dict() for a in AGENT_REGISTRY.values()]


@app.get("/tasks")
async def list_tasks(status: Optional[str] = None, agent: Optional[str] = None, limit: int = 50):
    if db_pool is None:
        return []
    if status and agent:
        rows = await db_fetch("SELECT * FROM tasks WHERE status=$1 AND agent_name=$2 ORDER BY created_at DESC LIMIT $3", status, agent, limit)
    elif status:
        rows = await db_fetch("SELECT * FROM tasks WHERE status=$1 ORDER BY created_at DESC LIMIT $2", status, limit)
    elif agent:
        rows = await db_fetch("SELECT * FROM tasks WHERE agent_name=$1 ORDER BY created_at DESC LIMIT $2", agent, limit)
    else:
        rows = await db_fetch("SELECT * FROM tasks ORDER BY created_at DESC LIMIT $1", limit)
    return [dict(r) for r in rows]




@app.get("/projects")
async def list_projects():
    if db_pool is None:
        return []
    rows = await db_fetch("SELECT * FROM projects ORDER BY created_at DESC")
    return [dict(r) for r in rows]


@app.post("/projects")
async def create_project(req: ProjectCreate):
    row = await db_fetchrow(
        "INSERT INTO projects (name, repo_url, metadata) VALUES ($1, $2, $3) RETURNING *",
        req.name, req.repo_url, json.dumps(req.metadata)
    )
    return dict(row)


@app.get("/logs")
async def get_logs(level: Optional[str] = None, limit: int = 100, offset: int = 0):
    if db_pool is None:
        return []
    if level:
        rows = await db_fetch("SELECT * FROM logs WHERE level=$1 ORDER BY timestamp DESC LIMIT $2 OFFSET $3", level, limit, offset)
    else:
        rows = await db_fetch("SELECT * FROM logs ORDER BY timestamp DESC LIMIT $1 OFFSET $2", limit, offset)
    return [dict(r) for r in rows]


@app.get("/analytics")
async def get_analytics():
    total_n = completed_n = failed_n = today_n = 0
    by_agent_rows: List[Dict[str, Any]] = []
    audit_total = audit_rejected = 0
    db_available = db_pool is not None

    if db_available:
        try:
            total = await db_fetchrow("SELECT COUNT(*) as n FROM tasks")
            success = await db_fetchrow("SELECT COUNT(*) as n FROM tasks WHERE status='completed'")
            failed = await db_fetchrow("SELECT COUNT(*) as n FROM tasks WHERE status='failed'")
            today = await db_fetchrow(
                "SELECT COUNT(*) as n FROM tasks WHERE created_at > NOW() - INTERVAL '24 hours'"
            )
            by_agent = await db_fetch(
                "SELECT agent_name, COUNT(*) as calls FROM tasks GROUP BY agent_name ORDER BY calls DESC LIMIT 10"
            )
            audit_stats = await db_fetchrow(
                "SELECT COUNT(*) as total, COUNT(*) FILTER (WHERE approved = false) as rejected FROM audit"
            )

            total_n = int(total["n"] or 0)
            completed_n = int(success["n"] or 0)
            failed_n = int(failed["n"] or 0)
            today_n = int(today["n"] or 0)
            by_agent_rows = [dict(r) for r in by_agent]
            if audit_stats:
                audit_total = int(audit_stats["total"] or 0)
                audit_rejected = int(audit_stats["rejected"] or 0)
        except Exception as e:
            logger.warning(f"Analytics DB fallback: {e}")
            db_available = False

    agents = [a.status_dict() for a in AGENT_REGISTRY.values()]
    total_agents = len(agents)
    active_agents = sum(1 for a in agents if a.get("status") == "running")
    error_agents = sum(1 for a in agents if a.get("status") == "error")
    security_agents = [a for a in agents if a.get("division") == "Security"]
    security_errors = sum(1 for a in security_agents if a.get("status") == "error")

    success_rate = round(completed_n / max(total_n, 1) * 100, 1) if total_n else 100.0
    agent_coverage = min(100.0, total_agents / 22 * 100) if total_agents else 0.0
    runtime_health = 100.0 - (error_agents / max(total_agents, 1) * 30)
    db_health = 100.0 if db_available else 75.0
    project_health = round(
        max(0.0, min(100.0, agent_coverage * 0.35 + runtime_health * 0.30 + success_rate * 0.25 + db_health * 0.10))
    )

    security_agent_health = 100.0
    if security_agents:
        security_agent_health = 100.0 - (security_errors / len(security_agents) * 35)
    audit_health = 100.0 - (audit_rejected / max(audit_total, 1) * 30) if audit_total else 100.0
    security_score = round(max(0.0, min(100.0, security_agent_health * 0.55 + audit_health * 0.30 + runtime_health * 0.15)))

    return {
        "total_tasks": total_n,
        "completed": completed_n,
        "failed": failed_n,
        "tasks_today": today_n,
        "success_rate": success_rate,
        "by_agent": by_agent_rows,
        "project_health": project_health,
        "security_score": security_score,
        "active_agents": active_agents,
        "total_agents": total_agents,
        "error_agents": error_agents,
        "db_available": db_available,
    }


@app.get("/memory/search")
async def memory_search(q: str = Query(...), collection: Optional[str] = None, limit: int = 5):
    results = await memory.search(q, collection, limit)
    return results


@app.post("/security/scan")
async def security_scan(req: SecurityScanRequest):
    agent = AGENT_REGISTRY.get("Cybersecurity")
    if agent:
        result = await agent.run(f"Security scan requested:\n\n{req.content}")
        return {"findings": result.output, "task_id": result.task_id}
    return {"findings": "Cybersecurity agent unavailable"}


@app.get("/audit")
async def get_audit(limit: int = 50):
    if db_pool is None:
        return []
    rows = await db_fetch("SELECT * FROM audit ORDER BY timestamp DESC LIMIT $1", limit)
    return [dict(r) for r in rows]


@app.get("/tools")
async def list_tools():
    return tool_registry.list_tools()


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Accept file upload and return its text content for agent context."""
    content = await file.read()
    try:
        text = content.decode("utf-8", errors="replace")
    except Exception:
        text = base64.b64encode(content).decode()
    return {
        "filename": file.filename,
        "size": len(content),
        "content": text[:8000],
        "truncated": len(text) > 8000,
    }


@app.get("/settings/notification-email")
async def get_notification_email():
    return {"email": NOTIFICATION_EMAIL}


@app.post("/settings/notification-email")
async def set_notification_email(req: NotificationEmailRequest):
    global NOTIFICATION_EMAIL
    NOTIFICATION_EMAIL = req.email
    return {"email": NOTIFICATION_EMAIL, "status": "saved"}


@app.get("/settings/ssh")
async def get_ssh_config():
    return {
        "host": SSH_HOST,
        "port": SSH_PORT,
        "user": SSH_USER,
        "configured": bool(SSH_HOST and SSH_USER),
    }


@app.post("/settings/ssh")
async def set_ssh_config(req: SSHConfigRequest):
    global SSH_HOST, SSH_PORT, SSH_USER, SSH_PASSWORD, SSH_KEY
    SSH_HOST = req.host
    SSH_PORT = req.port or 22
    SSH_USER = req.user
    SSH_PASSWORD = req.password or ""
    SSH_KEY = req.key or ""
    return {"status": "saved", "host": SSH_HOST, "user": SSH_USER}


@app.post("/settings/ssh/test")
async def test_ssh():
    result = await SSHTool().run({"command": "echo SSH_OK && uname -a && uptime"})
    return {"result": result, "ok": "SSH_OK" in result}


@app.post("/send-email")
async def send_email_endpoint(req: SendEmailRequest):
    result = await send_gmail(req.to, req.subject, req.body)
    return {"result": result}


@app.post("/send-email/test")
async def test_email():
    to = NOTIFICATION_EMAIL
    if not to:
        return {"result": "No notification email configured", "ok": False}
    result = await send_gmail(to, "✅ Jaxvora Email Test", "Your Jaxvora email notifications are working correctly!")
    return {"result": result, "ok": "sent" in result.lower()}


# === SECTION 10: WebSocket Endpoints ==========================================

@app.websocket("/ws/agents")
async def ws_agents(ws: WebSocket):
    await ws.accept()
    ws_manager.agents.add(ws)
    try:
        # Send current state immediately
        await ws.send_text(json.dumps({
            "type": "snapshot",
            "agents": [a.status_dict() for a in AGENT_REGISTRY.values()]
        }, default=str))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.agents.discard(ws)


@app.websocket("/ws/tasks")
async def ws_tasks(ws: WebSocket):
    await ws.accept()
    ws_manager.tasks_ws.add(ws)
    try:
        rows = await db_fetch("SELECT * FROM tasks ORDER BY created_at DESC LIMIT 20")
        await ws.send_text(json.dumps({"type": "snapshot", "tasks": [dict(r) for r in rows]}, default=str))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.tasks_ws.discard(ws)


@app.websocket("/ws/logs")
async def ws_logs(ws: WebSocket):
    await ws.accept()
    ws_manager.logs_ws.add(ws)
    try:
        rows = await db_fetch("SELECT * FROM logs ORDER BY timestamp DESC LIMIT 50")
        for r in reversed(rows):
            await ws.send_text(json.dumps({"type": "log", "level": r["level"], "message": r["message"], "ts": r["timestamp"].isoformat()}, default=str))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.logs_ws.discard(ws)


@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    ws_manager.chat_ws.add(ws)
    try:
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data).get("message", "")
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "message": "Invalid JSON payload."})
                continue
            if not msg:
                await ws.send_json({"type": "error", "message": "Message is required."})
                continue
            await ws.send_json({"type": "thinking", "message": "Orchestrator is planning..."})

            async def stream(event):
                await ws.send_json(event)

            result = await orchestrator.process(msg, stream_fn=stream)
            await ws.send_json({"type": "response", "message": result["response"], "plan": result["plan"], "agents": result["agents"]})
    except WebSocketDisconnect:
        ws_manager.chat_ws.discard(ws)
    except Exception as e:
        logger.warning(f"WebSocket chat error: {e}")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
        ws_manager.chat_ws.discard(ws)


# === SECTION 11: Startup / Shutdown Events ====================================

async def startup():
    global db_pool

    # Init DB
    if DATABASE_URL:
        try:
            db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10, command_timeout=30)
            async with db_pool.acquire() as conn:
                for stmt in SCHEMA.strip().split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        try:
                            await conn.execute(stmt)
                        except Exception as e:
                            logger.warning(f"Schema stmt skipped: {e}")
            logger.info("✓ PostgreSQL connected and schema ready")
        except Exception as e:
            logger.error(f"✗ PostgreSQL connection failed: {e}")
            db_pool = None
    else:
        logger.warning("DATABASE_URL not set — database features disabled")

    # Register agents and tools
    build_registry()
    tool_registry.register(FileSystemTool())
    tool_registry.register(TerminalTool())
    tool_registry.register(PostgreSQLTool())
    tool_registry.register(BrowserTool())
    tool_registry.register(SecurityScannerTool())
    tool_registry.register(CodeFormatterTool())
    tool_registry.register(EmailNotificationTool())
    tool_registry.register(SSHTool())

    logger.info(f"✓ {len(AGENT_REGISTRY)} agents registered")
    logger.info(f"✓ {len(tool_registry._tools)} MCP tools registered")

    # Announce ready
    if db_pool:
        try:
            await db_execute(
                "INSERT INTO logs (level, message) VALUES ('INFO', 'Jaxvora system_ready')"
            )
        except Exception:
            pass

    asyncio.create_task(_broadcast_ready())
    logger.info("🚀 Jaxvora ready")


async def _broadcast_ready():
    await asyncio.sleep(1)
    await ws_manager._send(ws_manager.agents, {"type": "system_ready", "ts": datetime.now().isoformat()})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
