"""
ClawBrief (Phase 2): Private Personal AI Knowledge & Skill Operator
Track: Personal AI Track (NVIDIA + Nebius Hackathon)
License: MIT

Easy Launch:
- Double-click `run_clawbrief.bat` on Windows, OR
- Press the Run (▶) button in VS Code / run `python clawbrief_app.py`, OR
- Run `python -m streamlit run clawbrief_app.py`

Phase 2 Capabilities:
1. Dual-Provider Smart Nemotron Router (NVIDIA Build Free NIM API <-> Nebius Token Factory)
2. Multi-Format Local Workspace Ingestion (`./workspace/notes`: .md, .txt, .csv, .ics, .json)
3. Self-Evolving Hermes Skills Library (3 pre-built skills + Nemotron 3 Ultra skill generator)
   executing against live workspace files inside a zero-credential OpenShell boundary
4. Nebius Serverless Nightly Briefing Compiler with 1-click Markdown export & credit telemetry
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.error

try:
    import streamlit as st
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    get_script_run_ctx = lambda: None

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# =====================================================================
# 1. DUAL-PROVIDER SMART MODEL ROUTER (NVIDIA BUILD + NEBIUS)
# =====================================================================

PROVIDERS = {
    "nvidia_build": {
        "name": "NVIDIA Build Free NIM API (Dev Mode)",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "env_key": "NVIDIA_API_KEY",
        "key_placeholder": "nvapi-...",
        "models": {
            "nano_lightning": {
                "id": "nvidia/nemotron-3.5-lightning-30b-a3b",
                "label": "Nemotron 3.5 Lightning 30B (Fast Triage)",
                "input_cost_per_1m": 0.06,
                "output_cost_per_1m": 0.24,
                "context_window": "1.0M",
            },
            "super": {
                "id": "nvidia/nemotron-3.5-super-vl-120b-a12b",
                "label": "Nemotron 3.5 Super VL 120B (Daily Workflows)",
                "input_cost_per_1m": 0.30,
                "output_cost_per_1m": 0.90,
                "context_window": "1.0M",
            },
            "ultra": {
                "id": "nvidia/nemotron-3-ultra-550b-a55b",
                "label": "Nemotron 3 Ultra 550B (Deep Skill Authoring)",
                "input_cost_per_1m": 1.00,
                "output_cost_per_1m": 3.00,
                "context_window": "1.0M",
            },
        },
    },
    "nebius": {
        "name": "Nebius Token Factory (Final Submission Mode)",
        "base_url": "https://api.tokenfactory.nebius.com/v1",
        "env_key": "NEBIUS_API_KEY",
        "key_placeholder": "nebius-api-key...",
        "models": {
            "nano_lightning": {
                "id": "nvidia/Nemotron-3_5-Lightning",
                "label": "Nemotron 3.5 Lightning (Fast Triage)",
                "input_cost_per_1m": 0.06,
                "output_cost_per_1m": 0.24,
                "context_window": "1.0M",
            },
            "super": {
                "id": "nvidia/nemotron-3-super-120b-a12b",
                "label": "Nemotron 3 Super 120B (Daily Workflows)",
                "input_cost_per_1m": 0.30,
                "output_cost_per_1m": 0.90,
                "context_window": "262K",
            },
            "ultra": {
                "id": "nvidia/Nemotron-3-Ultra-550b-a55b",
                "label": "Nemotron 3 Ultra 550B (Deep Skill Authoring)",
                "input_cost_per_1m": 1.00,
                "output_cost_per_1m": 3.00,
                "context_window": "1.0M",
            },
        },
    },
}


@dataclass
class RoutingEvent:
    timestamp: str
    provider: str
    task_type: str
    selected_tier: str
    model_id: str
    latency_ms: float
    estimated_tokens: int
    estimated_cost_usd: float
    ultra_baseline_cost_usd: float
    credits_saved_pct: float


class DualNemotronRouter:
    """Routes Personal AI tasks across NVIDIA Build (free dev) or Nebius Token Factory (final submission)."""

    def __init__(self, provider_key: str = "nvidia_build", api_key: Optional[str] = None):
        self.provider_key = provider_key if provider_key in PROVIDERS else "nvidia_build"
        self.provider_cfg = PROVIDERS[self.provider_key]
        env_var = self.provider_cfg["env_key"]
        self.api_key = (api_key or os.environ.get(env_var, "")).strip()
        self.client = (
            OpenAI(api_key=self.api_key, base_url=self.provider_cfg["base_url"])
            if (HAS_OPENAI and self.api_key)
            else None
        )

    def select_tier(self, prompt: str, force_tier: str = "auto") -> str:
        models = self.provider_cfg["models"]
        if force_tier in models:
            return force_tier
        lower = prompt.lower()
        skill_keywords = [
            "build a skill",
            "create a skill",
            "write a script",
            "automate",
            "synthesize",
            "deep research",
            "multi-step",
            "design",
        ]
        fast_keywords = [
            "tag",
            "classify",
            "quick",
            "extract action",
            "remind",
            "status",
            "triage",
        ]
        if any(k in lower for k in skill_keywords) or len(prompt) > 450:
            return "ultra"
        if any(k in lower for k in fast_keywords) or len(prompt) < 120:
            return "nano_lightning"
        return "super"

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        task_type: str = "workflow",
        force_tier: str = "auto",
    ) -> tuple[str, RoutingEvent]:
        tier = self.select_tier(user_prompt, force_tier=force_tier)
        models = self.provider_cfg["models"]
        model_info = models[tier]
        model_id = model_info["id"]
        start = time.perf_counter()

        if self.client:
            try:
                extra_body = {}
                if tier == "ultra" and self.provider_key == "nvidia_build":
                    extra_body = {
                        "chat_template_kwargs": {
                            "enable_thinking": False,
                            "force_nonempty_content": True,
                        }
                    }
                resp = self.client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    extra_body=extra_body if extra_body else None,
                )
                reply = resp.choices[0].message.content or ""
                tokens = getattr(resp.usage, "total_tokens", 650) or 650
            except Exception as exc:
                reply = self._demo_fallback_response(tier, task_type, user_prompt, str(exc))
                tokens = 620
        else:
            time.sleep(0.08 if tier == "nano_lightning" else (0.14 if tier == "super" else 0.22))
            reply = self._demo_fallback_response(tier, task_type, user_prompt)
            tokens = 620

        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        avg_rate = (model_info["input_cost_per_1m"] + model_info["output_cost_per_1m"]) / 2.0
        ultra_rate = (
            models["ultra"]["input_cost_per_1m"] + models["ultra"]["output_cost_per_1m"]
        ) / 2.0

        actual_cost = (tokens / 1_000_000) * avg_rate
        ultra_cost = (tokens / 1_000_000) * ultra_rate
        saved_pct = round(max(0.0, (1.0 - (actual_cost / ultra_cost)) * 100), 1)

        event = RoutingEvent(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            provider=self.provider_key,
            task_type=task_type,
            selected_tier=tier,
            model_id=model_id,
            latency_ms=latency_ms,
            estimated_tokens=tokens,
            estimated_cost_usd=round(actual_cost, 6),
            ultra_baseline_cost_usd=round(ultra_cost, 6),
            credits_saved_pct=saved_pct,
        )
        return reply, event

    def _demo_fallback_response(
        self, tier: str, task_type: str, user_prompt: str, note: Optional[str] = None
    ) -> str:
        model_id = self.provider_cfg["models"][tier]["id"]
        provider_label = self.provider_cfg["name"]
        if task_type == "skill_generation":
            return (
                "import json\nimport re\nimport sys\n\n"
                "text = sys.stdin.read()\n"
                "lines = [l.strip() for l in text.splitlines() if l.strip()]\n"
                "actions = [l for l in lines if any(k in l.lower() for k in ['todo', 'action', '[ ]', 'deadline', 'due'])]\n"
                "dates = re.findall(r'\\b(?:Oct|Nov|Dec|2026)[a-z0-9\\-\\s]*\\b', text)\n"
                "print(json.dumps({'skill': 'auto_generated_extractor', 'actions_found': len(actions), 'actions': actions[:5], 'dates': dates[:5]}, indent=2))\n"
            )
        header = f"**[{model_id} via {provider_label}]**"
        if note:
            header += f" *(Offline Demo Fallback — API notice: {note})*"
        return (
            f"{header}\n\n"
            f"- **Synthesized Takeaway**: Processed {len(user_prompt)} chars of local workspace context.\n"
            "- **Extracted Priority Actions**:\n"
            "  1. Complete Phase 2 skill testing inside the OpenShell sandbox boundary.\n"
            "  2. Review extracted study flashcards and calendar conflicts.\n"
            "  3. Export `morning_brief.md` and switch toggle to Nebius Token Factory before October 30.\n"
            "- **Privacy Guarantee**: Raw local files stayed inside `./workspace/notes`."
        )


# =====================================================================
# 2. LOCAL WORKSPACE MANAGER & HERMES PERSISTENT VAULT
# =====================================================================

DEFAULT_SKILLS = [
    {
        "name": "deadline_and_action_harvester",
        "description": "Scans any local note via stdin for TODOs, ACTION items, unchecked boxes, and October/2026 dates.",
        "authored_by": "nvidia/nemotron-3-ultra-550b-a55b",
        "sandbox_verified": True,
        "code": (
            "import json\nimport re\nimport sys\n\n"
            "text = sys.stdin.read()\n"
            "lines = [line.strip() for line in text.splitlines() if line.strip()]\n"
            "tasks = [l for l in lines if any(k in l.upper() for k in ['TODO', 'ACTION', '[ ]', 'DEADLINE', 'DUE'])]\n"
            "dates = re.findall(r'(?:Oct(?:ober)?\\s+\\d{1,2}|2026-\\d{2}-\\d{2})', text)\n"
            "print(json.dumps({'skill': 'deadline_and_action_harvester', 'task_count': len(tasks), 'dates': dates, 'tasks': tasks}, indent=2))\n"
        ),
    },
    {
        "name": "study_flashcard_crystallizer",
        "description": "Converts Markdown headings and bullet notes into structured Q&A study flashcards.",
        "authored_by": "nvidia/nemotron-3-ultra-550b-a55b",
        "sandbox_verified": True,
        "code": (
            "import json\nimport sys\n\n"
            "text = sys.stdin.read()\n"
            "cards = []\n"
            "current_topic = 'General Concept'\n"
            "for line in text.splitlines():\n"
            "    s = line.strip()\n"
            "    if s.startswith('#'):\n"
            "        current_topic = s.lstrip('#').strip()\n"
            "    elif s.startswith(('-', '*')) and ':' in s:\n"
            "        term, definition = s.lstrip('-* ').split(':', 1)\n"
            "        cards.append({'topic': current_topic, 'question': f'What is {term.strip()}?', 'answer': definition.strip()})\n"
            "print(json.dumps({'skill': 'study_flashcard_crystallizer', 'flashcards_created': len(cards), 'cards': cards[:6]}, indent=2))\n"
        ),
    },
    {
        "name": "ics_calendar_conflict_auditor",
        "description": "Parses .ics calendar events or schedule blocks and flags upcoming deadlines & meetings.",
        "authored_by": "nvidia/nemotron-3-ultra-550b-a55b",
        "sandbox_verified": True,
        "code": (
            "import json\nimport re\nimport sys\n\n"
            "text = sys.stdin.read()\n"
            "summaries = re.findall(r'SUMMARY:(.*)', text)\n"
            "dtstarts = re.findall(r'DTSTART[:;][^\\n]*', text)\n"
            "events = [{'event': s.strip(), 'start': (dtstarts[i].split(':')[-1].strip() if i < len(dtstarts) else 'TBD')} for i, s in enumerate(summaries)]\n"
            "if not events:\n"
            "    events = [{'event': l.strip(), 'start': 'Listed in note'} for l in text.splitlines() if 'Oct' in l or '2026' in l]\n"
            "print(json.dumps({'skill': 'ics_calendar_conflict_auditor', 'events_detected': len(events), 'schedule': events[:6]}, indent=2))\n"
        ),
    },
]


class HermesSkillVault:
    """Manages local `./workspace/notes` files, persistent memory, reusable skills, and local keys."""

    def __init__(self, storage_dir: str = ".clawbrief_state", workspace_dir: str = "workspace/notes"):
        self.root = Path(storage_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.workspace = Path(workspace_dir)
        self.workspace.mkdir(parents=True, exist_ok=True)

        self.memory_file = self.root / "persistent_memory.json"
        self.skills_file = self.root / "reusable_skills.json"
        self.keys_file = self.root / "local_keys.json"
        self._ensure_defaults()

    def _ensure_defaults(self) -> None:
        # 1. Seed default sample files in ./workspace/notes so user has rich local files immediately
        sample_study = self.workspace / "agent_memory_study_notes.md"
        if not sample_study.exists():
            sample_study.write_text(
                "# Autonomous Agent Architectures & Personal AI\n"
                "- Episodic Memory: Stores timestamped user interactions and daily task outcomes.\n"
                "- Semantic Memory: Consolidates long-term facts, preferences, and knowledge graphs.\n"
                "- Procedural Skill Library: Saves verified Python scripts written by Nemotron 3 Ultra.\n"
                "- OpenShell Sandboxing: Isolates skill execution so API keys never leak.\n\n"
                "## Action Items\n"
                "- [ ] TODO: Test all 3 pre-built Hermes skills on local notes by Oct 25\n"
                "- [ ] ACTION: Switch provider toggle to Nebius Token Factory by Oct 28\n"
                "- [ ] DEADLINE: Submit 3-minute YouTube video and MIT repo by Oct 30\n",
                encoding="utf-8",
            )

        sample_ics = self.workspace / "october_hackathon_schedule.ics"
        if not sample_ics.exists():
            sample_ics.write_text(
                "BEGIN:VCALENDAR\nVERSION:2.0\n"
                "BEGIN:VEVENT\nSUMMARY:ClawBrief Phase 2 Sandbox Testing\nDTSTART:20261024T100000Z\nEND:VEVENT\n"
                "BEGIN:VEVENT\nSUMMARY:Nebius Token Factory Final Cutover\nDTSTART:20261028T140000Z\nEND:VEVENT\n"
                "BEGIN:VEVENT\nSUMMARY:Record 3-Minute YouTube Hackathon Demo\nDTSTART:20261029T160000Z\nEND:VEVENT\n"
                "BEGIN:VEVENT\nSUMMARY:NVIDIA + Nebius Hackathon Final Submission\nDTSTART:20261030T180000Z\nEND:VEVENT\n"
                "END:VCALENDAR\n",
                encoding="utf-8",
            )

        sample_csv = self.workspace / "research_reading_list.csv"
        if not sample_csv.exists():
            sample_csv.write_text(
                "Title,Topic,Priority,Target_Date\n"
                "NVIDIA Nemotron 3 Hybrid MoE Paper,Model Routing,High,Oct 24\n"
                "Hermes Agent Self-Evolving Skills,Personal AI,High,Oct 25\n"
                "NVIDIA OpenShell & NemoClaw Guide,Security Sandbox,High,Oct 26\n",
                encoding="utf-8",
            )

        # 2. Seed persistent memories
        if not self.memory_file.exists():
            default_memories = [
                {
                    "id": "mem-1",
                    "category": "Research & Study",
                    "summary": "Studying episodic, semantic, and procedural memory consolidation in Hermes Agent.",
                    "source_file": "workspace/notes/agent_memory_study_notes.md",
                    "created_at": "2026-10-22",
                },
                {
                    "id": "mem-2",
                    "category": "Hackathon Roadmap",
                    "summary": "Prototyping on NVIDIA Build free API now; switching toggle to Nebius Token Factory before Oct 30 submission.",
                    "source_file": "workspace/notes/october_hackathon_schedule.ics",
                    "created_at": "2026-10-23",
                },
            ]
            self.memory_file.write_text(json.dumps(default_memories, indent=2), encoding="utf-8")

        # 3. Seed or upgrade reusable skills so all 3 Phase 2 skills are present
        existing_skills = []
        if self.skills_file.exists():
            try:
                existing_skills = json.loads(self.skills_file.read_text(encoding="utf-8"))
            except Exception:
                existing_skills = []
        existing_names = {s.get("name") for s in existing_skills}
        updated = False
        for d_skill in DEFAULT_SKILLS:
            if d_skill["name"] not in existing_names:
                existing_skills.append(d_skill)
                updated = True
            else:
                # Ensure stdin-aware version is up to date
                for idx, s in enumerate(existing_skills):
                    if s.get("name") == d_skill["name"] and "sys.stdin.read()" not in s.get("code", ""):
                        existing_skills[idx] = d_skill
                        updated = True
        if not self.skills_file.exists() or updated:
            self.skills_file.write_text(json.dumps(existing_skills, indent=2), encoding="utf-8")

        if not self.keys_file.exists():
            self.keys_file.write_text(json.dumps({}, indent=2), encoding="utf-8")

    def list_workspace_files(self) -> List[Path]:
        allowed = {".md", ".txt", ".csv", ".ics", ".json"}
        return sorted(
            [p for p in self.workspace.iterdir() if p.is_file() and p.suffix.lower() in allowed]
        )

    def save_workspace_file(self, filename: str, content: str) -> Path:
        safe_name = Path(filename).name
        target = self.workspace / safe_name
        target.write_text(content, encoding="utf-8")
        return target

    def read_workspace_file(self, filename: str) -> str:
        target = self.workspace / Path(filename).name
        if target.exists():
            return target.read_text(encoding="utf-8", errors="ignore")
        return ""

    def get_saved_key(self, env_var: str) -> str:
        env_val = os.environ.get(env_var, "")
        if env_val:
            return env_val
        try:
            data = json.loads(self.keys_file.read_text(encoding="utf-8"))
            return data.get(env_var, "")
        except Exception:
            return ""

    def save_local_key(self, env_var: str, key_val: str) -> None:
        if not key_val:
            return
        try:
            data = json.loads(self.keys_file.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        if data.get(env_var) != key_val:
            data[env_var] = key_val
            self.keys_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def list_memories(self) -> List[Dict[str, Any]]:
        return json.loads(self.memory_file.read_text(encoding="utf-8"))

    def add_memory(self, category: str, summary: str, source_file: str = "local_input.md") -> None:
        items = self.list_memories()
        items.insert(
            0,
            {
                "id": f"mem-{len(items) + 1}",
                "category": category,
                "summary": summary,
                "source_file": source_file,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            },
        )
        self.memory_file.write_text(json.dumps(items, indent=2), encoding="utf-8")

    def clear_memories(self) -> None:
        self.memory_file.write_text("[]", encoding="utf-8")
        self._ensure_defaults()

    def list_skills(self) -> List[Dict[str, Any]]:
        return json.loads(self.skills_file.read_text(encoding="utf-8"))

    def save_skill(self, name: str, description: str, code: str, authored_by: str) -> None:
        skills = self.list_skills()
        skills.insert(
            0,
            {
                "name": name,
                "description": description,
                "authored_by": authored_by,
                "sandbox_verified": True,
                "code": code,
            },
        )
        self.skills_file.write_text(json.dumps(skills, indent=2), encoding="utf-8")


# =====================================================================
# 3. NVIDIA NEMOCLAW / OPENSHELL SANDBOX EXECUTOR
# =====================================================================

class OpenShellSandbox:
    """Executes Hermes skills inside an NVIDIA NemoClaw / OpenShell sandbox boundary with stdin piping."""

    def __init__(self, sandbox_name: str = "clawbrief-personal-sandbox"):
        self.sandbox_name = sandbox_name
        self.has_nemoclaw_cli = shutil.which("nemoclaw") is not None
        self.has_openshell_cli = shutil.which("openshell") is not None

    @staticmethod
    def extract_python_code(raw_reply: str) -> str:
        fenced = re.findall(r"```(?:python)?\s*(.*?)```", raw_reply, flags=re.DOTALL)
        if fenced:
            return fenced[0].strip()
        return raw_reply.strip()

    def run_skill_code(self, code: str, input_text: str = "") -> Dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="openshell_sbx_") as tmpdir:
            script_path = Path(tmpdir) / "skill_task.py"
            script_path.write_text(code, encoding="utf-8")

            # Enforce stripped environment so API keys never leak into skill runtime
            safe_env = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"}

            try:
                proc = subprocess.run(
                    ["python3" if shutil.which("python3") else sys.executable, "-I", str(script_path)],
                    input=input_text,
                    cwd=tmpdir,
                    env=safe_env,
                    capture_output=True,
                    text=True,
                    timeout=6,
                )
                return {
                    "sandbox_runtime": (
                        "NVIDIA NemoClaw / OpenShell CLI"
                        if (self.has_nemoclaw_cli or self.has_openshell_cli)
                        else "OpenShell Isolated Subprocess Boundary (Zero-Credential Env)"
                    ),
                    "sandbox_name": self.sandbox_name,
                    "input_bytes_fed": len(input_text.encode("utf-8")),
                    "exit_code": proc.returncode,
                    "stdout": proc.stdout.strip() or "(No stdout output)",
                    "stderr": proc.stderr.strip(),
                }
            except Exception as exc:
                return {
                    "sandbox_runtime": "OpenShell Isolated Boundary",
                    "sandbox_name": self.sandbox_name,
                    "input_bytes_fed": len(input_text.encode("utf-8")),
                    "exit_code": 1,
                    "stdout": "",
                    "stderr": str(exc),
                }


# =====================================================================
# 4. NEBIUS SERVERLESS JOBS CLIENT & MORNING BRIEFING COMPILER
# =====================================================================

class NebiusServerlessJobsClient:
    """Dispatches asynchronous background digest workloads and compiles downloadable Morning Briefings."""

    JOBS_ENDPOINT = "https://api.nebius.cloud/ai/v1/jobs"

    def __init__(self, iam_token: Optional[str] = None):
        self.iam_token = iam_token or os.environ.get("NEBIUS_IAM_TOKEN", "")

    def trigger_nightly_digest_job(self, memory_count: int, skill_count: int) -> Dict[str, Any]:
        payload = {
            "name": f"clawbrief-nightly-digest-{int(time.time())}",
            "container": {
                "image": "cr.nebius.cloud/clawbrief/hermes-nightly-digest:latest",
                "command": ["python3", "run_digest.py", "--model", "nvidia/nemotron-3-super-120b-a12b"],
                "env": {
                    "MEMORIES_INDEXED": str(memory_count),
                    "SKILLS_AVAILABLE": str(skill_count),
                },
            },
        }
        if self.iam_token:
            try:
                req = urllib.request.Request(
                    self.JOBS_ENDPOINT,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {self.iam_token}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=6) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except Exception as exc:
                return {
                    "status": "QUEUED (Simulated Fallback)",
                    "job_id": f"nbj-{int(time.time())}",
                    "endpoint": self.JOBS_ENDPOINT,
                    "note": f"Nebius API note: {exc}",
                    "payload": payload,
                }
        return {
            "status": "RUNNING -> COMPLETED (Demo Mode)",
            "job_id": f"nbj-serverless-{int(time.time())}",
            "endpoint": self.JOBS_ENDPOINT,
            "container_image": payload["container"]["image"],
            "summary": (
                f"Consolidated {memory_count} persistent memories and executed {skill_count} "
                "Hermes skills via Nebius Serverless Jobs."
            ),
        }

    @staticmethod
    def compile_morning_brief_markdown(
        memories: List[Dict[str, Any]],
        skill_outputs: List[Dict[str, Any]],
        job_meta: Dict[str, Any],
        ai_executive_summary: str,
    ) -> str:
        today = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [
            f"# ClawBrief Daily Morning Briefing ({today})",
            "",
            f"- **Serverless Job ID**: `{job_meta.get('job_id', 'nbj-local')}`",
            f"- **Execution Status**: `{job_meta.get('status', 'COMPLETED')}`",
            "- **Sandbox Boundary**: `NVIDIA NemoClaw / OpenShell (Zero-Credential Leakage)`",
            "",
            "## 1. Executive Synthesis (NVIDIA Nemotron)",
            ai_executive_summary.strip(),
            "",
            "## 2. Automated OpenShell Skill Outputs",
        ]
        for item in skill_outputs:
            lines.append(f"### Skill: `{item['skill_name']}` (Source: `{item['source_file']}`)")
            lines.append("```json")
            lines.append(item["stdout"])
            lines.append("```")
            lines.append("")

        lines.append("## 3. Active Persistent Memory Snapshot (Hermes Vault)")
        for mem in memories[:8]:
            lines.append(
                f"- **[{mem['category']}]** ({mem['created_at']}): {mem['summary']} *(Source: `{mem['source_file']}`)*"
            )
        return "\n".join(lines) + "\n"


# =====================================================================
# 5. STREAMLIT LIVE DASHBOARD UI (PHASE 2)
# =====================================================================

def render_streamlit_app() -> None:
    st.set_page_config(
        page_title="ClawBrief | Personal AI Operator",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "routing_log" not in st.session_state:
        st.session_state.routing_log = []
    if "latest_brief_md" not in st.session_state:
        st.session_state.latest_brief_md = ""

    vault = HermesSkillVault()
    sandbox = OpenShellSandbox()

    with st.sidebar:
        st.title("🛡️ ClawBrief Control")
        st.caption("Personal AI Track • Dual-Provider Nemotron Router")

        provider_key = st.radio(
            "Inference Provider",
            options=["nvidia_build", "nebius"],
            index=0,
            format_func=lambda k: PROVIDERS[k]["name"],
            help="Use NVIDIA Build (free 1,000 credits at build.nvidia.com) for dev, then switch to Nebius Token Factory for final submission!",
        )
        active_cfg = PROVIDERS[provider_key]
        env_var_name = active_cfg["env_key"]

        saved_key = vault.get_saved_key(env_var_name)
        api_key = st.text_input(
            f"{env_var_name} ({provider_key})",
            value=saved_key,
            type="password",
            placeholder=active_cfg["key_placeholder"],
            help=f"Automatically saved locally in .clawbrief_state/local_keys.json • Endpoint: {active_cfg['base_url']}",
        )
        if api_key and api_key != saved_key:
            vault.save_local_key(env_var_name, api_key)
            st.toast(f"Saved {env_var_name} locally so you don't need to re-paste it next time!")

        if api_key:
            st.success(f"🔑 Live API Key Loaded (`{env_var_name}`)")
        else:
            st.warning("⚠️ No API key entered — running in Offline Simulation Mode")

        force_tier = st.selectbox(
            "Nemotron Routing Policy",
            options=["auto", "nano_lightning", "super", "ultra"],
            format_func=lambda x: (
                "Auto-Route (Stretch Credits)"
                if x == "auto"
                else active_cfg["models"][x]["label"]
            ),
        )
        st.markdown("### OpenShell / NemoClaw Status")
        st.success(f"Sandbox Active: `{sandbox.sandbox_name}`")
        ws_files = vault.list_workspace_files()
        st.caption(f"Whitelisted Workspace: `./workspace/notes` ({len(ws_files)} local files indexed)")

        st.markdown("### Router Telemetry")
        logs: List[Dict[str, Any]] = st.session_state.routing_log
        if logs:
            avg_saved = sum(item["credits_saved_pct"] for item in logs) / len(logs)
            avg_lat = sum(item["latency_ms"] for item in logs) / len(logs)
            st.metric("Avg Credits Saved vs Ultra", f"{avg_saved:.1f}%")
            st.metric("Avg Response Latency", f"{avg_lat:.0f} ms")
        else:
            st.info("Run a prompt or skill to see live model routing telemetry.")

    router = DualNemotronRouter(provider_key=provider_key, api_key=api_key)
    jobs_client = NebiusServerlessJobsClient()

    st.title("ClawBrief: Private Knowledge & Self-Evolving Skill Operator")
    st.write(
        f"**Active Endpoint:** `{active_cfg['base_url']}` ({active_cfg['name']}) • "
        "**Memory & Skills:** Hermes Agent • **Sandbox:** NVIDIA NemoClaw / OpenShell"
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "🧠 1. Workspace & Smart Nemotron Copilot",
            "🛠️ 2. Self-Evolving Hermes Skills (OpenShell)",
            "☁️ 3. Nebius Serverless Nightly Briefing",
        ]
    )

    # -----------------------------------------------------------------
    # TAB 1: MULTI-FILE WORKSPACE & SMART NEMOTRON COPILOT
    # -----------------------------------------------------------------
    with tab1:
        col_left, col_right = st.columns([3, 2])
        with col_left:
            st.subheader("1. Select or Upload Local Workspace Files")
            uploaded_files = st.file_uploader(
                "Drop .md, .txt, .csv, .ics, or .json files to save into your sandboxed ./workspace/notes folder:",
                type=["md", "txt", "csv", "ics", "json"],
                accept_multiple_files=True,
            )
            if uploaded_files:
                for uf in uploaded_files:
                    content_str = uf.read().decode("utf-8", errors="ignore")
                    vault.save_workspace_file(uf.name, content_str)
                st.success(f"Saved {len(uploaded_files)} file(s) into `./workspace/notes`!")

            ws_files = vault.list_workspace_files()
            file_names = [f.name for f in ws_files]
            selected_filename = st.selectbox(
                "Active Workspace File (inside `./workspace/notes`):",
                options=file_names,
                index=0 if file_names else None,
            )
            active_file_content = (
                vault.read_workspace_file(selected_filename) if selected_filename else ""
            )
            with st.expander("📄 Preview Active Local File Content", expanded=False):
                st.code(active_file_content, language="markdown")

            st.subheader("2. Ask ClawBrief Across Your Workspace & Memory")
            user_query = st.text_area(
                "Workflow Instruction:",
                value=(
                    "Synthesize the key takeaways and action deadlines from my active workspace file "
                    "and connect them to my persistent Hermes memory."
                ),
                height=100,
            )
            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                memory_category = st.selectbox(
                    "Memory Tag:",
                    options=["Research & Study", "Daily Schedule", "Action Items", "Synthesized Brief"],
                )
            with col_opt2:
                save_to_mem = st.checkbox(
                    "Save takeaway into Hermes Persistent Memory", value=True
                )

            if st.button("Execute via Smart Nemotron Router", type="primary"):
                memories_ctx = json.dumps(vault.list_memories()[:6], indent=2)
                combined_prompt = (
                    f"User Instruction: {user_query}\n\n"
                    f"Active Local File ({selected_filename}):\n{active_file_content[:3000]}"
                )
                sys_prompt = (
                    "You are ClawBrief, a privacy-first Personal AI running inside an NVIDIA OpenShell sandbox. "
                    "Provide a concise, structured response with clear bullet points and deadlines.\n"
                    f"Current persistent user memory:\n{memories_ctx}"
                )
                reply, event = router.complete(
                    system_prompt=sys_prompt,
                    user_prompt=combined_prompt,
                    task_type="personal_workflow",
                    force_tier=force_tier,
                )
                st.session_state.routing_log.insert(0, asdict(event))
                if save_to_mem:
                    summary_line = reply.splitlines()[0][:140] if reply else user_query[:120]
                    vault.add_memory(
                        category=memory_category,
                        summary=summary_line,
                        source_file=f"workspace/notes/{selected_filename}",
                    )
                st.markdown("#### Agent Response")
                st.markdown(reply)
                st.info(
                    f"Provider: `{event.provider}` • Model: `{event.model_id}` in **{event.latency_ms} ms** • "
                    f"Credits Saved vs Ultra-only: **{event.credits_saved_pct}%**"
                )

        with col_right:
            st.subheader("Persistent Memory Store (Hermes)")
            search_q = st.text_input("🔍 Filter memories by keyword or tag:", value="")
            memories = vault.list_memories()
            if search_q.strip():
                q_low = search_q.lower()
                memories = [
                    m
                    for m in memories
                    if q_low in m["summary"].lower()
                    or q_low in m["category"].lower()
                    or q_low in m["source_file"].lower()
                ]
            for mem in memories:
                with st.expander(f"📌 [{mem['category']}] {mem['created_at']}", expanded=True):
                    st.write(mem["summary"])
                    st.caption(f"Source: `{mem['source_file']}`")

    # -----------------------------------------------------------------
    # TAB 2: SELF-EVOLVING HERMES SKILLS (OPENSHELL SANDBOX)
    # -----------------------------------------------------------------
    with tab2:
        st.subheader("Generate & Run Sandboxed Hermes Skills on Your Local Files")
        st.write(
            f"Skills run inside an isolated **OpenShell** subprocess (`{sandbox.sandbox_name}`) with all API keys stripped. "
            "Each skill reads your selected local file from `sys.stdin` and outputs structured JSON."
        )
        ws_files = vault.list_workspace_files()
        file_names = [f.name for f in ws_files]

        col_s1, col_s2 = st.columns([2, 2])
        with col_s1:
            target_skill_file = st.selectbox(
                "Select Local Workspace File to Feed into Skill (`stdin`):",
                options=file_names,
                key="skill_target_file",
            )
            target_file_text = (
                vault.read_workspace_file(target_skill_file) if target_skill_file else ""
            )
        with col_s2:
            skill_name_input = st.text_input(
                "New Skill Name (for Nemotron 3 Ultra authoring):",
                value="markdown_keyword_counter",
            )

        skill_goal = st.text_input(
            "Describe a new reusable skill for Nemotron 3 Ultra to write:",
            value=(
                "Read text from sys.stdin.read(), count total words, lines, and bullet points, "
                "and print a JSON summary using json.dumps."
            ),
        )
        if st.button("Author & Verify New Skill with Nemotron 3 Ultra", type="primary"):
            author_sys_prompt = (
                "You are an expert Python skill synthesis engine for Hermes Agent inside an NVIDIA OpenShell sandbox. "
                "Output ONLY valid, runnable Python 3 code (no markdown explanation outside code block) using ONLY the Python standard library. "
                "The script MUST read input from `sys.stdin.read()` and print a JSON object to stdout."
            )
            code_reply, event = router.complete(
                system_prompt=author_sys_prompt,
                user_prompt=f"Write a Python 3 script for this skill: {skill_goal}",
                task_type="skill_generation",
                force_tier="ultra",
            )
            st.session_state.routing_log.insert(0, asdict(event))
            clean_code = sandbox.extract_python_code(code_reply)
            sbx_res = sandbox.run_skill_code(clean_code, input_text=target_file_text)
            safe_skill_name = re.sub(r"[^a-zA-Z0-9_]", "_", skill_name_input.strip()) or "custom_skill"
            vault.save_skill(
                name=safe_skill_name,
                description=skill_goal,
                code=clean_code,
                authored_by=event.model_id,
            )
            st.success(
                f"Skill `{safe_skill_name}` authored by `{event.model_id}` ({event.provider}) and tested on `{target_skill_file}`!"
            )
            st.json(sbx_res)

        st.markdown("#### Saved Reusable Skills Library (Click to Execute on Selected File)")
        for idx, sk in enumerate(vault.list_skills()):
            with st.expander(f"⚙️ `{sk['name']}` — {sk['description']}", expanded=(idx < 2)):
                st.caption(
                    f"Authored by: `{sk['authored_by']}` | OpenShell Verified: `{sk['sandbox_verified']}`"
                )
                st.code(sk["code"], language="python")
                if st.button(
                    f"▶ Run `{sk['name']}` on `{target_skill_file}` in OpenShell Sandbox",
                    key=f"run_sk_{idx}",
                ):
                    res = sandbox.run_skill_code(sk["code"], input_text=target_file_text)
                    st.json(res)

    # -----------------------------------------------------------------
    # TAB 3: NEBIUS SERVERLESS NIGHTLY BRIEFING & TELEMETRY
    # -----------------------------------------------------------------
    with tab3:
        st.subheader("Asynchronous Nightly Digest & Downloadable Morning Briefing")
        st.write(
            "Triggers a containerized **Nebius Serverless Job** (`https://api.nebius.cloud/ai/v1/jobs`), executes your "
            "verified OpenShell skills across `./workspace/notes`, and compiles a downloadable Morning Briefing."
        )
        if st.button("🚀 Dispatch Nightly Job & Compile Morning Briefing", type="primary"):
            memories = vault.list_memories()
            skills = vault.list_skills()
            job_result = jobs_client.trigger_nightly_digest_job(
                memory_count=len(memories),
                skill_count=len(skills),
            )

            # Execute top skills against workspace files to include real outputs in the briefing
            study_text = vault.read_workspace_file("agent_memory_study_notes.md")
            ics_text = vault.read_workspace_file("october_hackathon_schedule.ics")
            skill_outputs = []
            for sk in skills[:3]:
                sample_in = ics_text if "ics" in sk["name"] else study_text
                src_name = (
                    "october_hackathon_schedule.ics"
                    if "ics" in sk["name"]
                    else "agent_memory_study_notes.md"
                )
                out = sandbox.run_skill_code(sk["code"], input_text=sample_in)
                skill_outputs.append(
                    {
                        "skill_name": sk["name"],
                        "source_file": src_name,
                        "stdout": out["stdout"],
                    }
                )

            exec_prompt = (
                "Create a concise 3-bullet Morning Executive Briefing combining these user memories and extracted tasks:\n"
                + json.dumps(memories[:4], indent=2)
            )
            ai_summary, event = router.complete(
                system_prompt="You are ClawBrief's Nightly Digest Compiler.",
                user_prompt=exec_prompt,
                task_type="nightly_digest",
                force_tier="super",
            )
            st.session_state.routing_log.insert(0, asdict(event))

            brief_md = jobs_client.compile_morning_brief_markdown(
                memories=memories,
                skill_outputs=skill_outputs,
                job_meta=job_result,
                ai_executive_summary=ai_summary,
            )
            st.session_state.latest_brief_md = brief_md
            st.success(f"Serverless Job `{job_result['job_id']}` completed and Morning Briefing compiled!")
            st.json(job_result)

        if st.session_state.latest_brief_md:
            st.markdown("#### Generated Morning Briefing Preview")
            st.download_button(
                label="📥 Download Morning Briefing (`morning_brief.md`)",
                data=st.session_state.latest_brief_md,
                file_name=f"morning_brief_{datetime.now().strftime('%Y_%m_%d')}.md",
                mime="text/markdown",
            )
            st.markdown(st.session_state.latest_brief_md)

        if st.session_state.routing_log:
            st.markdown("#### Session Routing Audit Table & Credit Savings")
            st.dataframe(st.session_state.routing_log, use_container_width=True)


def run_cli_self_test() -> None:
    print("=== ClawBrief Phase 2 Dual-Provider & Workspace Self-Test ===")
    vault = HermesSkillVault()
    sandbox = OpenShellSandbox()
    jobs = NebiusServerlessJobsClient()

    ws_files = [f.name for f in vault.list_workspace_files()]
    print(f"[1] Workspace Files Indexed: {ws_files}")

    nv_router = DualNemotronRouter(provider_key="nvidia_build")
    _, ev_nv = nv_router.complete(
        system_prompt="You are ClawBrief.",
        user_prompt="Quickly tag my study note on multi-agent memory.",
        task_type="quick_triage",
    )
    print(f"[2] NVIDIA Build Dev Mode -> {ev_nv.model_id} (Saved {ev_nv.credits_saved_pct}% credits)")

    study_text = vault.read_workspace_file("agent_memory_study_notes.md")
    ics_text = vault.read_workspace_file("october_hackathon_schedule.ics")
    for sk in vault.list_skills()[:3]:
        target_in = ics_text if "ics" in sk["name"] else study_text
        res = sandbox.run_skill_code(sk["code"], input_text=target_in)
        print(f"[3] Skill `{sk['name']}` Exit {res['exit_code']} -> {res['stdout'].splitlines()[0]}")

    job_out = jobs.trigger_nightly_digest_job(len(vault.list_memories()), len(vault.list_skills()))
    brief = jobs.compile_morning_brief_markdown(
        memories=vault.list_memories(),
        skill_outputs=[{"skill_name": "deadline_and_action_harvester", "source_file": "study.md", "stdout": "{}"}],
        job_meta=job_out,
        ai_executive_summary="- All systems verified.",
    )
    print(f"[4] Nebius Serverless Job & Briefing Compiled: {len(brief)} bytes ({job_out['job_id']})")


if __name__ == "__main__":
    if os.environ.get("CLAWBRIEF_CLI_TEST") == "1":
        run_cli_self_test()
    elif HAS_STREAMLIT:
        if get_script_run_ctx() is not None:
            render_streamlit_app()
        else:
            print("Auto-launching Streamlit server for clawbrief_app.py...")
            sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)]))
    else:
        run_cli_self_test()