```mermaid
%%{init: {"flowchart": {"htmlLabels": true, "curve": "basis"}, "theme": "dark", "themeVariables": {"primaryColor": "#1a1b26", "primaryTextColor": "#a9b1d6", "primaryBorderColor": "#565f89", "lineColor": "#7aa2f7", "tertiaryColor": "#24283b"}}}%%
flowchart TB
    subgraph User["User"]
        U["👤 User Input"]
    end

    subgraph Orchestrator["Chief Orchestrator — v1.0 TAOR Protocol"]
        direction TB
        RP["🧠 Route / Intent\nAttachment? Gmail? SSH? Doctor? Align?"]
        subgraph Loop["Think → Dispatch → Act → Observe → Reflect → Final"]
            THINK["💭 THINK\nLLM reasons, extracts subtasks,\nchecks risk_flags, sets confidence"]
            DISPATCH["📤 DISPATCH\nInvoke sub-agent\nwith context + output schema"]
            ACT["🔧 ACT\nExecute MCP Tool\nor process agent result"]
            OBSERVE["👁️ OBSERVE\nProcess result,\nupdate context"]
            REFLECT["🔄 REFLECT\nGoal fulfilled?\nQuality check?"]
            CONFIRM["⚠️ CONFIRMATION GATE\nRisk detected → ask user"]
            FINAL["✅ FINAL\nSynthesize v1.0 answer\n(iteration, confidence, agents)"]
        end
        RP -->|General query| THINK
        THINK -->|dispatch agent| DISPATCH --> ACT --> OBSERVE --> REFLECT
        REFLECT -->|continue| THINK
        REFLECT -->|done| FINAL
        THINK -->|risk_flags found| CONFIRM -->|user approves| THINK
        CONFIRM -->|user denies| FINAL
        FINAL --> U
    end

    subgraph Divisions["6 Divisions · 37 Agents"]
        direction TB

        subgraph ENG["🟢 Engineering · 10 agents"]
            direction LR
            AIEng["AI Engineer\n⚡ deepseek_v4"]
            SWEng["Software Engineer\n⚡ deepseek_v4"]
            Debug["Debug Agent\n⚡ deepseek_v4"]
            QA["QA/Test Agent\n💠 deepseek"]
            CR["Code Review\n🔵 groq"]
            Arch["Architecture\n⚡ deepseek_v4\n🏁 Division Lead"]
            DB["Database\n💠 deepseek"]
            DevOps["DevOps\n💠 deepseek"]
            BE["Backend Engineer\n💠 deepseek"]
            FE["Frontend Engineer\n💠 deepseek"]
        end

        subgraph SEC["🔴 Security · 6 agents"]
            direction LR
            Cyber["Cybersecurity\n⚡ deepseek_v4\n🏁 Division Lead"]
            Red["Red Team\n💠 deepseek"]
            Compl["Compliance\n💠 deepseek"]
            Vuln["Vulnerability Scanner\n💠 deepseek"]
            IAM["Auth & IAM Agent\n💠 deepseek"]
            NetSec["Network Security\n💠 deepseek"]
        end

        subgraph DATA["🟡 Data · 6 agents"]
            direction LR
            DA["Data Analyst\n💠 deepseek"]
            BI["BI Agent\n💠 deepseek"]
            DE["Data Engineer\n💠 deepseek\n🏁 Division Lead"]
            ML["ML Engineer\n⚡ deepseek_v4"]
            ETL["ETL Engineer\n💠 deepseek"]
            RAG["RAG Specialist\n⚡ deepseek_v4"]
        end

        subgraph CAREER["🟣 Career · 5 agents"]
            direction LR
            Resume["Resume Agent\n💠 deepseek"]
            IC["Interview Coach\n💠 deepseek"]
            CC["Career Coach\n💠 deepseek\n🏁 Division Lead"]
            JobSearch["Job Search Agent\n💠 deepseek"]
            AppTracker["Application Tracker\n💠 deepseek"]
        end

        subgraph PROD["🟠 Product · 5 agents"]
            direction LR
            PM["Product Manager\n💠 deepseek\n🏁 Division Lead"]
            Docs["Documentation\n💠 deepseek"]
            Research["Research\n💠 deepseek"]
            UX["UX Designer\n💠 deepseek"]
            ReqAnalyst["Requirements Analyst\n💠 deepseek"]
        end

        subgraph EXEC["⚪ Executive · 4 agents"]
            PI["Project Intelligence\n🔵 groq"]
            Doctor["Jaxvora Doctor\n⚡ deepseek_v4"]
            Strategy["Strategy Agent\n⚡ deepseek_v4"]
            RiskPlan["Risk & Planning\n⚡ deepseek_v4\n🏁 Division Lead"]
        end
    end

    subgraph Tools["MCP Tools · permission-gated"]
        FS["📁 FileSystemTool\n🟡 medium"]
        TERM["🖥️ TerminalTool\n🔴 high"]
        PG["🗄️ PostgreSQLTool\n🔴 high"]
        BR["🌐 BrowserTool\n🟢 low"]
        SECTOOL["🔍 SecurityScanner\n🟢 low"]
        CODE["✨ CodeFormatter\n🟢 low"]
        EMAIL["📧 EmailNotification\n🟡 medium"]
        GMAIL["📬 GmailAutomation\n⛔ critical"]
        SSH["🔐 SSHTool\n⛔ critical"]
        WEB["🔎 WebSearchTool\n🟢 low"]
        INVOKE["🤝 AgentInvokeTool\n🟡 medium"]
        SOCIAL["📱 SocialPost\n⛔ critical"]
        CRUNNER["⚡ CodeRunner\n🟡 medium"]
        PW["🎭 Playwright\n🟡 medium"]
        FPREV["🖼️ FrontendPreview\n🟢 low"]
        SRUNNER["🚀 ServerRunner\n🟡 medium"]
    end

    subgraph RAG["RAG Engine"]
        EMB["🧬 FastEmbed\nbge-small-en-v1.5"]
        CHUNK["✂️ Chunker"]
        HYBRID["🔀 Hybrid Search"]
    end

    subgraph LLM["LLM Providers"]
        GROQ["Groq\nllama-3.3-70b"]
        OR_V3["OpenRouter\nDeepSeek V3"]
        OR_V4["OpenRouter\nDeepSeek V4 Flash Free"]
    end

    subgraph External["External"]
        DDG["DuckDuckGo"]
        GMAIL_API["Gmail API"]
        SSH_VM["Linode VM"]
    end

    subgraph DB["PostgreSQL (Neon) · 12 tables"]
        TASKS["tasks"]
        AUDIT["audit"]
        AGENT_HIST["agent_history"]
        KB["knowledge_base"]
        RAG_DOCS["rag_documents"]
        SESSIONS["jaxvora_sessions"]
        SUBTASK_LOG["jaxvora_subtask_log"]
        OP_LOG["jaxvora_operation_log"]
        SSH_AUDIT["jaxvora_ssh_audit"]
    end

    %% Agent Collaboration (network edges)
    Arch <-.->|Division Lead| AIEng & SWEng & Debug & QA & CR & DB & DevOps & BE & FE
    Cyber <-.->|Division Lead| Red & Compl & Vuln & IAM & NetSec
    DE <-.->|Division Lead| DA & BI & ML & ETL & RAG
    CC <-.->|Division Lead| Resume & IC & JobSearch & AppTracker
    PM <-.->|Division Lead| Docs & Research & UX & ReqAnalyst
    RiskPlan <-.->|Division Lead| PI & Doctor & Strategy

    %% Cross-division collaboration
    SWEng -.-> QA & CR & DevOps & FE
    AIEng -.-> ML & DE & RAG
    Arch -.-> DevOps & DB & PM
    Cyber -.-> Red & Compl & Vuln & NetSec
    PM -.-> Docs & Research & Arch & UX
    Doctor -.->|self-heal| all other agents
    Strategy -.-> PI & RiskPlan

    %% Tool access with permissions
    THINK -.-> DISPATCH -.-> ACT
    ACT --> INVOKE
    INVOKE -.->|calls any agent| Divisions
    ACT --> FS & TERM & PG & BR & SECTOOL & CODE & EMAIL & GMAIL & SSH & WEB & SOCIAL & CRUNNER & PW & FPREV & SRUNNER
    WEB --> DDG
    GMAIL --> GMAIL_API
    SSH --> SSH_VM

    %% LLM routing
    AIEng & SWEng & Debug & Arch & Cyber & ML & RAG & Doctor & Strategy & RiskPlan --> OR_V4
    QA & DB & DevOps & Red & Compl & DA & BI & DE & ETL & Resume & IC & CC & JobSearch & AppTracker --> OR_V3
    BE & FE & Vuln & IAM & NetSec & UX & ReqAnalyst --> OR_V3
    CR & PI & ChiefOrchestrator --> GROQ

    %% RAG augmentation
    Loop --> RAG
    RAG --> EMB & CHUNK & HYBRID
    HYBRID <--> RAG_DOCS

    %% DB
    Loop --> DB
    Divisions --> DB

    classDef user fill:#1a1b26,stroke:#bb9af7,stroke-width:2px
    classDef orchestrator fill:#24283b,stroke:#7aa2f7,stroke-width:3px
    classDef loop fill:#1a1b26,stroke:#7aa2f7,stroke-width:2px
    classDef eng fill:#1a1b26,stroke:#9ece6a,stroke-width:1.5px
    classDef sec fill:#1a1b26,stroke:#f7768e,stroke-width:1.5px
    classDef data fill:#1a1b26,stroke:#e0af68,stroke-width:1.5px
    classDef career fill:#1a1b26,stroke:#bb9af7,stroke-width:1.5px
    classDef prod fill:#1a1b26,stroke:#ff9e64,stroke-width:1.5px
    classDef exec fill:#1a1b26,stroke:#a9b1d6,stroke-width:1.5px
    classDef tool fill:#24283b,stroke:#73daca,stroke-width:1px
    classDef llm fill:#1a1b26,stroke:#f7768e,stroke-width:1px
    classDef rag fill:#1a1b26,stroke:#e0af68,stroke-width:1px
    classDef db fill:#1a1b26,stroke:#565f89,stroke-width:1px

    class U user
    class Orchestrator,RP orchestrator
    class THINK,DISPATCH,ACT,OBSERVE,REFLECT,CONFIRM,FINAL,Loop loop
    class AIEng,SWEng,Debug,QA,CR,Arch,DB,DevOps,BE,FE eng
    class Cyber,Red,Compl,Vuln,IAM,NetSec sec
    class DA,BI,DE,ML,ETL,RAG data
    class Resume,IC,CC,JobSearch,AppTracker career
    class PM,Docs,Research,UX,ReqAnalyst prod
    class PI,Doctor,Strategy,RiskPlan exec
    class FS,TERM,PG,BR,SECTOOL,CODE,EMAIL,GMAIL,SSH,WEB,INVOKE,SOCIAL,CRUNNER,PW,FPREV,SRUNNER tool
    class GROQ,OR_V3,OR_V4 llm
    class EMB,CHUNK,HYBRID,RAG rag
    class TASKS,AUDIT,AGENT_HIST,KB,RAG_DOCS,SESSIONS,SUBTASK_LOG,OP_LOG,SSH_AUDIT db
```
