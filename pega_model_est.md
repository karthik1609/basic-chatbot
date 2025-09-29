```mermaid
flowchart LR
    A[End user<br/>(Customer/Agent/Dev)] -->|asks question / triggers case| B[Pega App (UI/Case)]
    B --> C[Pega GenAI Orchestrator]
    C --> D[Guardrails & Policies<br/>(prompt templates, allow/deny, audit)]
    C --> E{Which capability?}
    E -->|Knowledge Q&A| F[GenAI Knowledge Buddy]
    E -->|Dev assist / test / blueprint| G[GenAI for Builders<br/>(blueprint, code suggestions,<br/>auto-tests)]
    E -->|Ops/Decisioning| H[Pega Decision Hub / Cases]

    %% Knowledge Buddy (RAG) path
    F --> I[Content Connectors & Index<br/>(sharepoint, web, files)]
    I --> J[Retrieval<br/>(search + chunk + rerank)]
    J --> K[LLM Call via Pega GenAI services<br/>(provider: Azure OpenAI, etc.)]
    D --> K
    K --> L[Grounded Answer + Citations]
    L --> B
    L --> M[Telemetry & Audit<br/>(prompts, sources, outcomes)]

    %% Decision/Action path
    H --> N[Policies/Next Best Action<br/>(predictive/prescriptive)]
    N --> O[Automations & Cases<br/>(assign, update, RPA, integrations)]
    O --> B

    %% Dev path
    G --> P[Scaffolding & Tests Generated]
    P --> O
    M -.-> D
```