flowchart LR
  %% === Personas / Entry ===
  U[Researcher] -->|goals & constraints| ORCH

  %% === Orchestrator & Loop ===
  subgraph Control["AI-Scientist Orchestrator"]
    ORCH["Orchestrator<br/>(run config, provenance)"]
    PLAN["Planner / Hypothesis Gen"]
    EMS["Experiment Manager<br/>(branching, depth, retries)"]
    EXEC["Executor / Runner"]
    CRIT["Auto-Review / Critic"]
    OBS["Telemetry & Cost<br/>(tokens, GPU time, budgets)"]

    ORCH --> PLAN --> EMS --> EXEC --> CRIT
    CRIT --> PLAN
    OBS -.-> PLAN
    OBS -.-> EXEC
  end

  %% === MCP Fabric ===
  subgraph MCPFabric["MCP Integration Layer"]
    direction TB
    MCPc["MCP Client (in agents)"]
    MCPb["MCP Broker / Router"]
    S1["(MCP Server) Code Exec<br/>Python / Conda / Docker"]
    S2["(MCP Server) FS & Artifacts<br/>(read/write, datasets)"]
    S3["(MCP Server) Web/Search<br/>(arXiv, GitHub, docs)"]
    S4["(MCP Server) Cluster Control<br/>(K8s / Slurm / Batch)"]
    S5["(MCP Server) PDF/LaTeX & Review<br/>(citations, refs)"]
    MCPc --> MCPb
    MCPb --> S1 & S2 & S3 & S4 & S5
  end

  %% === Execution & Data ===
  subgraph Exec["Execution & Data"]
    direction TB
    Q[(Run Queue)]
    ART["Artifact Store<br/>(datasets, models, logs, PDFs)"]
    METRICS["Metrics / Plots / Tables"]
    Q --> EXEC
    EXEC -->|logs & results| ART
    METRICS -.-> OBS
  end

  %% === Models / Tools ===
  subgraph Models["Model Pool / Tools"]
    direction TB
    MSEL["Model Selector"]
    M1[(Frontier LLM A)]
    M2[(Frontier LLM B)]
    M3[(Coder LLM)]
    VLM[(Vision / Multimodal)]
    TOOL[(Specialized Tools)]
    MSEL --> M1 & M2 & M3 & VLM
  end

  %% === Data / Configs ===
  CFG["Templates & Prompts<br/>(policies, guardrails)"]
  REFS["Literature / Refs Cache"]
  DSETS["Datasets Registry"]

  %% === Cross-edges ===
  ORCH --> CFG
  PLAN --> REFS
  EMS --> Q
  EMS -. metrics .-> METRICS
  CRIT -. figures/tables .-> METRICS

  %% Orchestrator/Agents <-> MCP
  ORCH -. tool calls .-> MCPc
  PLAN -. data/refs .-> MCPc
  EMS -. spawn/scale .-> MCPc
  CRIT -. PDF review .-> MCPc

  %% MCP servers used during runs
  EXEC -->|exec code| S1
  EXEC -->|read/write| S2
  CRIT -->|web fetch| S3
  EXEC -->|dispatch jobs| S4
  CRIT -->|LaTeX/PDF| S5

  %% Models usage
  PLAN --> MSEL
  EMS --> MSEL

  %% Feedback loops
  ART --> OBS
  CRIT -->|accept/reject| EMS

  %% User loop
  CRIT -->|final PDF & supplements| U[Researcher]
