```mermaid
```mermaid
flowchart LR
    A[End user\n(Customer/Agent/Dev)] -->|asks question / triggers case| B[Pega App (UI/Case)]
    B --> C[Pega GenAI Orchestrator]
    C --> D[Guardrails & Policies\n(prompt templates, allow/deny, audit)]
    C --> E{Which capability?}
    E -->|Knowledge Q&A| F[GenAI Knowledge Buddy]
    E -->|Dev assist / test / blueprint| G[GenAI for Builders\n(blueprint, code suggestions,\nauto-tests)]
    E -->|Ops/Decisioning| H[Pega Decision Hub / Cases]

    %% Knowledge Buddy (RAG) path
    F --> I[Content Connectors & Index\n(sharepoint, web, files)]
    I --> J[Retrieval\n(search + chunk + rerank)]
    J --> K[LLM Call via Pega GenAI services\n(provider: Azure OpenAI, etc.)]
    D --> K
    K --> L[Grounded Answer + Citations]
    L --> B
    L --> M[Telemetry & Audit\n(prompts, sources, outcomes)]

    %% Decision/Action path
    H --> N[Policies/Next Best Action\n(predictive/prescriptive)]
    N --> O[Automations & Cases\n(assign, update, RPA, integrations)]
    O --> B

    %% Dev path
    G --> P[Scaffolding & Tests Generated]
    P --> O
    M -.-> D
```