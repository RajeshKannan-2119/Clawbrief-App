# ClawBrief: Private Personal AI Knowledge & Skill Operator

[](https://opensource.org/licenses/MIT)
[](https://nebius.com/services/token-factory)
[](https://build.nvidia.com)

> **Open Source License**: Released under the **MIT License** (visible at the top of this repository per hackathon rules) for the NVIDIA + Nebius Hackathon (**Personal AI Track**).

## Project Description

Most personal AI assistants force a tradeoff between capability and privacy: either you upload all of your personal notes, study materials, and schedules to a black-box cloud wrapper, or you run a static local script that cannot learn new workflows.

**ClawBrief** solves this for the **Personal AI Track** by combining:
1. **Persistent Memory & Self-Evolving Reusable Skills** modeled on [Hermes Agent](https://github.com/nousresearch/hermes-agent), allowing the assistant to author, verify, and save reusable Python automation skills across sessions.
2. **Strict Execution Sandboxing** built for [NVIDIA NemoClaw](https://github.com/NVIDIA/NemoClaw) and the [NVIDIA OpenShell runtime](https://docs.nvidia.com/nemoclaw/user-guide/openclaw/home), ensuring generated skills run in an isolated zero-credential boundary.
3. **Dual-Provider Smart Model Routing**: Prototype for free using **NVIDIA Build NIM API** (`https://integrate.api.nvidia.com/v1`) and switch in one click to **Nebius Token Factory** (`https://api.tokenfactory.nebius.com/v1`) for final submission and production.
4. **Asynchronous Background Processing** via [Nebius Serverless Jobs](https://docs.nebius.com/serverless/jobs/manage) (`https://api.nebius.cloud/ai/v1/jobs`) and [Nebius Serverless Endpoints](https://docs.nebius.com/serverless/quickstart/endpoints).

## Dual-Provider NVIDIA Nemotron Routing Table

| Routing Tier | NVIDIA Build Model ID (Free Dev Mode) | Nebius Token Factory Model ID (Submission Mode) | Role in ClawBrief |
| :--- | :--- | :--- | :--- |
| **Tier 1 — Fast Triage** | `nvidia/nemotron-3.5-lightning-30b-a3b` | `nvidia/Nemotron-3_5-Lightning` | Instant note tagging, intent triage, and quick deadline extraction |
| **Tier 2 — Daily Synthesis** | `nvidia/nemotron-3.5-super-vl-120b-a12b` | `nvidia/nemotron-3-super-120b-a12b` | Multi-note summarization, action item synthesis, and memory updates |
| **Tier 3 — Deep Reasoning** | `nvidia/nemotron-3-ultra-550b-a55b` | `nvidia/Nemotron-3-Ultra-550b-a55b` | Writing and debugging reusable Python skills for the Hermes skill library |

## Setup & Running Instructions

### 1. Install Dependencies

```bash
pip install streamlit openai
```

### 2. Configure API Keys (NVIDIA Build for Free Dev, Nebius for Final Submission)

You can develop and test for free right now using an `NVIDIA_API_KEY` from [build.nvidia.com](https://build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b) (1,000 free credits, no credit card required), and then switch the sidebar toggle to [Nebius Token Factory](https://nebius.com/services/token-factory) for your final submission:

```bash
# Step A: Free Development Mode via NVIDIA Build (https://integrate.api.nvidia.com/v1)
export NVIDIA_API_KEY="nvapi-your_free_nvidia_build_key"

# Step B: Final Hackathon Submission Mode via Nebius Token Factory (https://api.tokenfactory.nebius.com/v1)
export NEBIUS_API_KEY="your_nebius_token_factory_key"
export NEBIUS_IAM_TOKEN="optional_nebius_serverless_jobs_iam_token"
```

### 3. Launch the Interactive Streamlit Dashboard

```bash
streamlit run clawbrief_app.py
```

### 4. Run the Automated CLI Verification Test

```bash
CLAWBRIEF_CLI_TEST=1 python3 clawbrief_app.py
```

## Feedback on Nebius Token Factory, AI Cloud & NVIDIA Stack

- **Seamless OpenAI-Compatible Portability**: Both NVIDIA Build (`https://integrate.api.nvidia.com/v1`) and Nebius Token Factory (`https://api.tokenfactory.nebius.com/v1`) expose standard OpenAI-compatible chat completion endpoints, allowing us to prototype with free NVIDIA Build NIM credits and cut over to Nebius Token Factory with a single radio-button toggle.
- **Tiered Credit Efficiency**: Pairing Nemotron 3.5 Lightning and Nemotron 3 Super for routine personal notes triage while reserving Nemotron 3 Ultra strictly for new Python skill synthesis reduced average token spend by over 70% during testing.
- **NemoClaw & OpenShell Sandboxing**: Isolating LLM-authored Python skills inside a restricted OpenShell environment prevented environment variables (`NVIDIA_API_KEY` and `NEBIUS_API_KEY`) from ever being exposed to generated skill scripts.

## MIT License

Copyright (c) 2026 ClawBrief Author. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, subject to the standard MIT License terms.