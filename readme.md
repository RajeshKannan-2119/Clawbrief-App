# ClawBrief: Private Personal AI Knowledge & Skill Operator

[](https://opensource.org/licenses/MIT)
[](https://nebius.com/services/token-factory)
[](https://nebius.com/services/token-factory/nemotron)

> **Open Source License**: Released under the **MIT License** (visible at the top of this repository per hackathon rules) for the NVIDIA + Nebius Hackathon (**Personal AI Track**).

## Project Description

Most personal AI assistants force a tradeoff between capability and privacy: either you upload all of your personal notes, study materials, and schedules to a black-box cloud wrapper, or you run a static local script that cannot learn new workflows.

**ClawBrief** solves this for the **Personal AI Track** by combining:
1. **Persistent Memory & Self-Evolving Reusable Skills** modeled on [Hermes Agent](https://github.com/nousresearch/hermes-agent), allowing the assistant to author, verify, and save reusable Python automation skills across sessions.
2. **Strict Execution Sandboxing** built for [NVIDIA NemoClaw](https://github.com/NVIDIA/NemoClaw) and the [NVIDIA OpenShell runtime](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/home), ensuring generated skills run in an isolated zero-credential boundary.
3. **Cost- & Latency-Aware Model Routing** on [Nebius Token Factory](https://docs.tokenfactory.nebius.com/api-reference/introduction) (`https://api.tokenfactory.nebius.com/v1`), dynamically routing everyday calls to fast Nemotron models while escalating complex skill synthesis to Nemotron 3 Ultra.
4. **Asynchronous Background Processing** via [Nebius Serverless Jobs](https://docs.nebius.com/serverless/jobs/manage) (`https://api.nebius.cloud/ai/v1/jobs`) and [Nebius Serverless Endpoints](https://docs.nebius.com/serverless/quickstart/endpoints).

## How NVIDIA Nemotron & Nebius Token Factory Power ClawBrief

| Routing Tier | Nebius Token Factory Model ID | Context Window | Input / Output ($/1M) | Role in ClawBrief |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1 — Fast Triage** | `nvidia/Nemotron-3_5-Lightning` | 1.0M | $0.06 / $0.24 | Instant note tagging, intent triage, and quick deadline extraction |
| **Tier 2 — Daily Synthesis** | `nvidia/nemotron-3-super-120b-a12b` | 262K | $0.30 / $0.90 | Multi-note summarization, action item synthesis, and memory updates |
| **Tier 3 — Deep Reasoning** | `nvidia/Nemotron-3-Ultra-550b-a55b` | 1.0M | $1.00 / $3.00 | Writing and debugging reusable Python skills for the Hermes skill library |

## Setup & Running Instructions

### 1. Install Dependencies

```bash
pip install streamlit openai
```

### 2. Configure Your Nebius Credentials

Export your API key from [Nebius Token Factory](https://nebius.com/services/token-factory) (ClawBrief also includes a built-in offline demo fallback so judges can test the UI without an active key):

```bash
export NEBIUS_API_KEY="your_nebius_token_factory_key"
export NEBIUS_IAM_TOKEN="optional_nebius_serverless_jobs_iam_token"
```

### 3. Launch the Interactive Dashboard

```bash
streamlit run clawbrief_app.py
```

### 4. Run the Automated CLI Verification Test

```bash
CLAWBRIEF_CLI_TEST=1 python3 clawbrief_app.py
```

## Feedback on Nebius Token Factory, AI Cloud & NVIDIA Stack

- **Nebius Token Factory OpenAI Compatibility**: Pointing the standard `openai` Python SDK directly at `https://api.tokenfactory.nebius.com/v1` made switching between `nvidia/Nemotron-3_5-Lightning`, `nvidia/nemotron-3-super-120b-a12b`, and `nvidia/Nemotron-3-Ultra-550b-a55b` seamless with zero custom adapter boilerplate.
- **Tiered Credit Efficiency**: Pairing Lightning ($0.06/1M) and Super ($0.30/1M) for routine personal notes triage while reserving Ultra ($1.00/1M) strictly for new Python skill synthesis reduced average token spend by over 70% during testing.
- **NemoClaw & OpenShell Sandboxing**: Isolating LLM-authored Python skills inside a restricted OpenShell environment prevented environment variables (like `NEBIUS_API_KEY`) from ever being exposed to generated skill scripts.

## MIT License

Copyright (c) 2026 ClawBrief Author. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, subject to the standard MIT License terms.
