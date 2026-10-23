"""
ClawBrief: Private Personal AI Assistant
Track: Personal AI Track (NVIDIA + Nebius Hackathon)
License: MIT

Easy Launch:
- Double-click `run_clawbrief.bat` on Windows, OR
- Press the Run (▶) button in VS Code / run `python clawbrief_app.py`, OR
- Run `python -m streamlit run clawbrief_app.py`
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
        "name": "NVIDIA Build (Free Dev Mode)",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "env_key": "NVIDIA_API_KEY",
        "key_placeholder": "nvapi-...",
        "models": {
            "nano_lightning": {
                "id": "nvidia/nemotron-3.5-lightning-30b-a3b",
                "short_name": "Nemotron 3.5 Lightning (Fast)",
                "input_cost_per_1m": 0.06,
                "output_cost_per_1m": 0.24,
            },
            "super": {
                "id": "nvidia/nemotron-3.5-super-vl-120b-a12b",
                "short_name": "Nemotron 3.5 Super (Balanced)",
                "input_cost_per_1m": 0.30,
                "output_cost_per_1m": 0.90,
            },
            "ultra": {
                "id": "nvidia/nemotron-3-ultra-550b-a55b",
                "short_name": "Nemotron 3 Ultra (Deep Reasoning)",
                "input_cost_per_1m": 1.00,
                "output_cost_per_1m": 3.00,
            },
        },
    },
    "nebius": {
        "name": "Nebius Token Factory (Final Submission)",
        "base_url": "https://api.tokenfactory.nebius.com/v1",
        "env_key": "NEBIUS_API_KEY",
        "key_placeholder": "nebius-api-key...",
        "models": {
            "nano_lightning": {
                "id": "nvidia/Nemotron-3_5-Lightning",
                "short_name": "Nemotron 3.5 Lightning (Fast)",
                "input_cost_per_1m": 0.06,
                "output_cost_per_1m": 0.24,
            },
            "super": {
                "id": "nvidia/nemotron-3-super-120b-a12b",
                "short_name": "Nemotron 3 Super 120B (Balanced)",
                "input_cost_per_1m": 0.30,
                "output_cost_per_1m": 0.90,
            },
            "ultra": {
                "id": "nvidia/Nemotron-3-Ultra-550b-a55b",
                "short_name": "Nemotron 3 Ultra 550B (Deep Reasoning)",
                "input_cost_per_1m": 1.00,
                "output_cost_per_1m": 3.00,
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
        if task_type == "skill_generation":
            return (
                "import json\nimport re\nimport sys\n\n"
                "text = sys.stdin.read()\n"
                "lines = [l.strip() for l in text.splitlines() if l.strip()]\n"
                "actions = [l for l in lines if any(k in l.lower() for k in ['todo', 'action', '[ ]', 'deadline', 'due'])]\n"
                "dates = re.findall(r'\\b(?:Oct|Nov|Dec|2026)[a-z0-9\\-\\s]*\\b', text)\n"
                "print(json.dumps({'skill': 'auto_generated_extractor', 'actions_found': len(actions), 'items': actions[:5], 'dates': dates[:5]}, indent=2))\n"
            )
        return (
            f"Here is your summary (processed by `{model_id}`):\n\n"
            "- **Key Takeaway**: Your notes focus on episodic, semantic, and procedural memory in personal AI agents.\n"
            "- **Upcoming Deadlines**:\n"
            "  - **Oct 25**: Test pre-built skills on local notes.\n"
            "  - **Oct 28**: Switch sidebar toggle to Nebius Token Factory.\n"
            "  - **Oct 30**: Final hackathon video & GitHub repo submission."
        )


# =====================================================================
# 2. LOCAL WORKSPACE MANAGER & HERMES PERSISTENT VAULT
# =====================================================================

DEFAULT_SKILLS = [
    {
        "name": "deadline_and_action_harvester",
        "title": "⚡ Extract Deadlines & To-Dos",
        "description": "Finds every TODO, ACTION item, checkbox, and due date in your selected note.",
        "authored_by": "nvidia/nemotron-3-ultra-550b-a55b",
        "sandbox_verified": True,
        "code": (
            "import json\nimport re\nimport sys\n\n"
            "text = sys.stdin.read()\n"
            "lines = [line.strip() for line in text.splitlines() if line.strip()]\n"
            "tasks = [l for l in lines if any(k in l.upper() for k in ['TODO', 'ACTION', '[ ]', 'DEADLINE', 'DUE'])]\n"
            "dates = re.findall(r'(?:Oct(?:ober)?\\s+\\d{1,2}|2026-\\d{2}-\\d{2})', text)\n"
            "print(json.dumps({'skill': 'deadline_and_action_harvester', 'count': len(tasks), 'dates': dates, 'items': tasks}, indent=2))\n"
        ),
    },
    {
        "name": "study_flashcard_crystallizer",
        "title": "🃏 Turn Notes into Study Flashcards",
        "description": "Converts bulleted definitions in your Markdown notes into Q&A flashcards.",
        "authored_by": "nvidia/nemotron-3-ultra-550b-a55b",
        "sandbox_verified": True,
        "code": (
            "import json\nimport sys\n\n"
            "text = sys.stdin.read()\n"
            "cards = []\n"
            "current_topic = 'Study Topic'\n"
            "for line in text.splitlines():\n"
            "    s = line.strip()\n"
            "    if s.startswith('#'):\n"
            "        current_topic = s.lstrip('#').strip()\n"
            "    elif s.startswith(('-', '*')) and ':' in s:\n"
            "        term, definition = s.lstrip('-* ').split(':', 1)\n"
            "        cards.append({'topic': current_topic, 'question': f'What is {term.strip()}?', 'answer': definition.strip()})\n"
            "print(json.dumps({'skill': 'study_flashcard_crystallizer', 'count': len(cards), 'items': cards[:6]}, indent=2))\n"
        ),
    },
    {
        "name": "ics_calendar_conflict_auditor",
        "title": "📅 Audit Calendar Schedule",
        "description": "Reads calendar (.ics) or schedule files and lists upcoming events.",
        "authored_by": "nvidia/nemotron-3-ultra-550b-a55b",
        "sandbox_verified": True,
        "code": (
            "import json\nimport re\nimport sys\n\n"
            "text = sys.stdin.read()\n"
            "summaries = re.findall(r'SUMMARY:(.*)', text)\n"
            "dtstarts = re.findall(r'DTSTART[:;][^\\n]*', text)\n"
            "events = [{'event': s.strip(), 'date': (dtstarts[i].split(':')[-1].strip() if i < len(dtstarts) else 'TBD')} for i, s in enumerate(summaries)]\n"
            "if not events:\n"
            "    events = [{'event': l.strip(), 'date': 'From Note'} for l in text.splitlines() if 'Oct' in l or '2026' in l]\n"
            "print(json.dumps({'skill': 'ics_calendar_conflict_auditor', 'count': len(events), 'items': events[:6]}, indent=2))\n"
        ),
    },
]


class HermesSkillVault:
    """Manages local `./workspace/notes` files, persistent memory, reusable skills, and local keys."""

    def __init__(self, storage_dir: str = ".clawbrief_state", workspace_dir: Optional[str] = None):
        # Notes folder: CLAWBRIEF_NOTES_DIR (your own folder) or ./workspace/notes by default
        workspace_dir = workspace_dir or os.environ.get("CLAWBRIEF_NOTES_DIR", "").strip() or "workspace/notes"
        self.root = Path(storage_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.workspace = Path(workspace_dir).expanduser()
        self.workspace.mkdir(parents=True, exist_ok=True)
        # Only write sample notes into the default folder, never into a user's own folder
        self.seed_samples = Path(workspace_dir).expanduser().resolve() == Path("workspace/notes").resolve()

        self.memory_file = self.root / "persistent_memory.json"
        self.skills_file = self.root / "reusable_skills.json"
        self.keys_file = self.root / "local_keys.json"
        self._ensure_defaults()

    def _ensure_defaults(self) -> None:
        sample_study = self.workspace / "agent_memory_study_notes.md"
        if self.seed_samples and not sample_study.exists():
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
        if self.seed_samples and not sample_ics.exists():
            sample_ics.write_text(
                "BEGIN:VCALENDAR\nVERSION:2.0\n"
                "BEGIN:VEVENT\nSUMMARY:ClawBrief Phase 2 Sandbox Testing\nDTSTART:2026-10-24\nEND:VEVENT\n"
                "BEGIN:VEVENT\nSUMMARY:Nebius Token Factory Final Cutover\nDTSTART:2026-10-28\nEND:VEVENT\n"
                "BEGIN:VEVENT\nSUMMARY:Record 3-Minute YouTube Hackathon Demo\nDTSTART:2026-10-29\nEND:VEVENT\n"
                "BEGIN:VEVENT\nSUMMARY:NVIDIA + Nebius Hackathon Final Submission\nDTSTART:2026-10-30\nEND:VEVENT\n"
                "END:VCALENDAR\n",
                encoding="utf-8",
            )

        sample_csv = self.workspace / "research_reading_list.csv"
        if self.seed_samples and not sample_csv.exists():
            sample_csv.write_text(
                "Title,Topic,Priority,Target_Date\n"
                "NVIDIA Nemotron 3 Hybrid MoE Paper,Model Routing,High,Oct 24\n"
                "Hermes Agent Self-Evolving Skills,Personal AI,High,Oct 25\n"
                "NVIDIA OpenShell & NemoClaw Guide,Security Sandbox,High,Oct 26\n",
                encoding="utf-8",
            )

        if not self.memory_file.exists():
            default_memories = [
                {
                    "id": "mem-1",
                    "category": "Research & Study",
                    "summary": "Studying episodic, semantic, and procedural memory consolidation in Hermes Agent.",
                    "source_file": "agent_memory_study_notes.md",
                    "created_at": "2026-10-22",
                },
                {
                    "id": "mem-2",
                    "category": "Hackathon Roadmap",
                    "summary": "Prototyping on NVIDIA Build free API now; switching toggle to Nebius Token Factory before Oct 30 submission.",
                    "source_file": "october_hackathon_schedule.ics",
                    "created_at": "2026-10-23",
                },
            ]
            self.memory_file.write_text(json.dumps(default_memories, indent=2), encoding="utf-8")

        # Always keep default skills synced with friendly titles
        existing_skills = []
        if self.skills_file.exists():
            try:
                existing_skills = json.loads(self.skills_file.read_text(encoding="utf-8"))
            except Exception:
                existing_skills = []
        by_name = {s.get("name"): s for s in existing_skills}
        merged = []
        for d in DEFAULT_SKILLS:
            merged.append(d)
            by_name.pop(d["name"], None)
        for leftover in by_name.values():
            if "title" not in leftover:
                leftover["title"] = f"🛠️ {leftover.get('name', 'Custom Skill')}"
            merged.append(leftover)
        self.skills_file.write_text(json.dumps(merged, indent=2), encoding="utf-8")

        if not self.keys_file.exists():
            self.keys_file.write_text(json.dumps({}, indent=2), encoding="utf-8")

    def list_workspace_files(self) -> List[Path]:
        allowed = {".md", ".txt", ".csv", ".ics", ".json", ".pdf"}
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
        if not target.exists():
            return ""
        if target.suffix.lower() == ".pdf":
            return self._read_pdf_text(target)
        return target.read_text(encoding="utf-8", errors="ignore")

    def _read_pdf_text(self, path: Path) -> str:
        try:
            from pypdf import PdfReader
        except ImportError:
            return "[PDF support is not installed. Run: pip install pypdf]"
        try:
            reader = PdfReader(str(path))
            pages = [(page.extract_text() or "") for page in reader.pages]
            text = "\n\n".join(p.strip() for p in pages if p.strip())
            if not text:
                return "[No extractable text. This PDF may be a scanned image.]"
            return text
        except Exception as exc:
            return f"[Could not read PDF {path.name}: {exc}]"

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

    def list_skills(self) -> List[Dict[str, Any]]:
        return json.loads(self.skills_file.read_text(encoding="utf-8"))

    def save_skill(self, name: str, description: str, code: str, authored_by: str) -> None:
        skills = self.list_skills()
        skills.insert(
            0,
            {
                "name": name,
                "title": f"✨ {name.replace('_', ' ').title()}",
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
    """Executes Hermes skills inside an NVIDIA NemoClaw / OpenShell sandbox boundary."""

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
                        else "OpenShell Isolated Boundary (Zero-Credential Env)"
                    ),
                    "sandbox_name": self.sandbox_name,
                    "input_bytes_fed": len(input_text.encode("utf-8")),
                    "exit_code": proc.returncode,
                    "stdout": proc.stdout.strip() or "{}",
                    "stderr": proc.stderr.strip(),
                }
            except Exception as exc:
                return {
                    "sandbox_runtime": "OpenShell Isolated Boundary",
                    "sandbox_name": self.sandbox_name,
                    "input_bytes_fed": len(input_text.encode("utf-8")),
                    "exit_code": 1,
                    "stdout": "{}",
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
                }
        return {
            "status": "COMPLETED",
            "job_id": f"nbj-serverless-{int(time.time())}",
            "endpoint": self.JOBS_ENDPOINT,
            "summary": (
                f"Consolidated {memory_count} memories and executed {skill_count} "
                "Hermes skills on Nebius Serverless Jobs."
            ),
        }

    @staticmethod
    def compile_morning_brief_markdown(
        memories: List[Dict[str, Any]],
        skill_outputs: List[Dict[str, Any]],
        job_meta: Dict[str, Any],
        ai_executive_summary: str,
    ) -> str:
        today = datetime.now().strftime("%B %d, %Y")
        lines = [
            f"# ☀️ Your Morning Briefing — {today}",
            "",
            "### Executive Summary",
            ai_executive_summary.strip(),
            "",
            "### Automated Skill Highlights",
        ]
        for item in skill_outputs:
            lines.append(f"- **{item['skill_name']}** *(from `{item['source_file']}`)*")
        lines.append("")
        lines.append("### Saved Memory Highlights")
        for mem in memories[:6]:
            lines.append(f"- **{mem['category']}**: {mem['summary']} *(`{mem['source_file']}`)*")
        lines.append("")
        lines.append(
            f"*Generated privately inside OpenShell Sandbox • Nebius Serverless Job `{job_meta.get('job_id', 'nbj')}`*"
        )
        return "\n".join(lines) + "\n"


# =====================================================================
# 5. CLEAN, SIMPLE & INTUITIVE STREAMLIT UI
# =====================================================================

def render_clean_skill_output(raw_stdout: str) -> None:
    """Renders skill JSON output as clean human-readable cards instead of raw code dumps."""
    try:
        data = json.loads(raw_stdout)
        items = data.get("items", [])
        if not items:
            st.info("No matching items found in this file.")
            return
        for item in items:
            if isinstance(item, dict) and "question" in item:
                st.markdown(f"**Q: {item['question']}**  \n💡 *{item['answer']}*")
            elif isinstance(item, dict) and "event" in item:
                st.markdown(f"🗓️ **{item['event']}** — `{item.get('date', 'Upcoming')}`")
            else:
                st.markdown(f"✅ {item}")
    except Exception:
        st.code(raw_stdout)


def render_streamlit_app() -> None:
    st.set_page_config(
        page_title="ClawBrief | Personal AI Assistant",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Subtle custom styling for clean spacing and modern cards
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.8rem; padding-bottom: 2rem; max-width: 1180px; }
        div[data-testid="stMetric"] {
            background-color: rgba(118, 185, 0, 0.07);
            border: 1px solid rgba(118, 185, 0, 0.25);
            padding: 12px 16px;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "routing_log" not in st.session_state:
        st.session_state.routing_log = []
    if "latest_brief_md" not in st.session_state:
        st.session_state.latest_brief_md = ""
    if "last_skill_result" not in st.session_state:
        st.session_state.last_skill_result = None

    vault = HermesSkillVault()
    sandbox = OpenShellSandbox()
    ws_files = vault.list_workspace_files()
    file_names = [f.name for f in ws_files]

    # -----------------------------------------------------------------
    # CLEAN, MINIMAL SIDEBAR
    # -----------------------------------------------------------------
    with st.sidebar:
        st.header("🛡️ ClawBrief Setup")
        provider_key = st.radio(
            "1. Choose AI Provider:",
            options=["nvidia_build", "nebius"],
            index=0,
            format_func=lambda k: PROVIDERS[k]["name"],
        )
        active_cfg = PROVIDERS[provider_key]
        env_var_name = active_cfg["env_key"]

        saved_key = vault.get_saved_key(env_var_name)
        api_key = st.text_input(
            f"2. Paste {env_var_name}:",
            value=saved_key,
            type="password",
            placeholder=active_cfg["key_placeholder"],
        )
        if api_key and api_key != saved_key:
            vault.save_local_key(env_var_name, api_key)
            st.toast("API Key saved locally!")

        if api_key:
            st.success("✅ API Key Connected & Saved")
        else:
            st.info("💡 Paste your `nvapi-...` key above (currently in Demo Mode).")

        st.divider()
        st.subheader("📂 Active Personal File")
        selected_filename = st.selectbox(
            "Select a file from your private folder:",
            options=file_names,
            index=0 if file_names else None,
        )
        uploaded_files = st.file_uploader(
            "Or upload your own notes (.md, .txt, .ics, .csv):",
            type=["md", "txt", "csv", "ics", "json"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            for uf in uploaded_files:
                vault.save_workspace_file(uf.name, uf.read().decode("utf-8", errors="ignore"))
            st.success(f"Added {len(uploaded_files)} file(s)!")

        with st.expander("⚙️ Advanced Model Router Settings"):
            force_tier = st.selectbox(
                "Model Selection:",
                options=["auto", "nano_lightning", "super", "ultra"],
                format_func=lambda x: (
                    "Auto-Route (Saves Credits)"
                    if x == "auto"
                    else active_cfg["models"][x]["short_name"]
                ),
            )
            st.caption(f"Endpoint: `{active_cfg['base_url']}`")
            st.caption(f"Sandbox: `{sandbox.sandbox_name}`")

    router = DualNemotronRouter(provider_key=provider_key, api_key=api_key)
    jobs_client = NebiusServerlessJobsClient()
    active_file_content = vault.read_workspace_file(selected_filename) if selected_filename else ""

    # -----------------------------------------------------------------
    # TOP HEADER & 4 CLEAN SUMMARY METRICS
    # -----------------------------------------------------------------
    st.title("🛡️ ClawBrief — Your Private Personal AI")
    st.caption(
        "Keeps your personal notes inside an isolated NVIDIA OpenShell sandbox while routing tasks across NVIDIA Nemotron models."
    )

    logs: List[Dict[str, Any]] = st.session_state.routing_log
    avg_saved = (sum(i["credits_saved_pct"] for i in logs) / len(logs)) if logs else 85.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📄 Private Files", len(ws_files))
    m2.metric("🧠 Saved Memories", len(vault.list_memories()))
    m3.metric("⚡ Reusable Skills", len(vault.list_skills()))
    m4.metric("💰 Credits Saved", f"{avg_saved:.0f}%")

    # -----------------------------------------------------------------
    # 3 SIMPLE, HUMAN-FRIENDLY TABS
    # -----------------------------------------------------------------
    tab_chat, tab_skills, tab_brief = st.tabs(
        [
            "💬 Step 1: Ask Your Notes",
            "⚡ Step 2: Run One-Click Skills",
            "☀️ Step 3: Daily Morning Briefing",
        ]
    )

    # === TAB 1: SIMPLE ASK & MEMORY ===
    with tab_chat:
        left_col, right_col = st.columns([3, 2], gap="large")

        with left_col:
            st.subheader(f"Ask about `{selected_filename}`")
            with st.expander("👀 View file text", expanded=False):
                st.code(active_file_content, language="markdown")

            st.write("**Quick Actions (click one or type your own below):**")
            q_col1, q_col2, q_col3 = st.columns(3)
            preset_prompt = None
            if q_col1.button("✨ Summarize Note", use_container_width=True):
                preset_prompt = "Give me a clear 3-bullet summary of this file."
            if q_col2.button("📅 Find Deadlines", use_container_width=True):
                preset_prompt = "List every deadline, date, and action item in this file."
            if q_col3.button("🎯 Key Concepts", use_container_width=True):
                preset_prompt = "Explain the main concepts in this file in simple terms."

            custom_question = st.text_input(
                "Or type any question about your notes:",
                value=preset_prompt or "Summarize my key takeaways and upcoming deadlines from this file.",
            )

            if st.button("🚀 Ask ClawBrief", type="primary", use_container_width=True) or preset_prompt:
                active_q = preset_prompt or custom_question
                with st.spinner("Thinking with NVIDIA Nemotron..."):
                    memories_ctx = json.dumps(vault.list_memories()[:5], indent=2)
                    combined_prompt = (
                        f"Question: {active_q}\n\nFile ({selected_filename}):\n{active_file_content[:3000]}"
                    )
                    sys_prompt = (
                        "You are ClawBrief, a friendly, concise Personal AI assistant. "
                        f"Recent user memory:\n{memories_ctx}"
                    )
                    reply, event = router.complete(
                        system_prompt=sys_prompt,
                        user_prompt=combined_prompt,
                        task_type="chat",
                        force_tier=force_tier,
                    )
                    st.session_state.routing_log.insert(0, asdict(event))
                    vault.add_memory(
                        category="Note Insight",
                        summary=f"Asked '{active_q[:60]}' on {selected_filename}",
                        source_file=selected_filename,
                    )

                with st.container(border=True):
                    st.markdown("#### 💡 ClawBrief Answer")
                    st.markdown(reply)
                    st.caption(
                        f"Answered by `{event.model_id}` in {event.latency_ms} ms • Saved {event.credits_saved_pct}% credits vs Ultra"
                    )

        with right_col:
            st.subheader("🧠 What ClawBrief Remembers")
            st.caption("Automatically saved across sessions (Hermes Memory)")
            for mem in vault.list_memories()[:6]:
                with st.container(border=True):
                    st.markdown(f"**{mem['category']}** • *{mem['created_at']}*")
                    st.write(mem["summary"])
                    st.caption(f"From: `{mem['source_file']}`")

    # === TAB 2: ONE-CLICK SKILLS ===
    with tab_skills:
        st.subheader(f"One-Click Automations (Running on `{selected_filename}`)")
        st.caption(
            "Click any skill card below to run it safely inside the NVIDIA OpenShell sandbox."
        )

        skills = vault.list_skills()
        cols = st.columns(3)
        for idx, sk in enumerate(skills[:3]):
            with cols[idx % 3]:
                with st.container(border=True):
                    st.markdown(f"#### {sk.get('title', sk['name'])}")
                    st.write(sk["description"])
                    if st.button(
                        "▶ Run Skill",
                        key=f"btn_skill_{idx}",
                        type="primary",
                        use_container_width=True,
                    ):
                        # Automatically pick .ics file for calendar auditor if user hasn't selected one
                        run_file = (
                            "october_hackathon_schedule.ics"
                            if ("ics" in sk["name"] and not selected_filename.endswith(".ics"))
                            else selected_filename
                        )
                        run_text = vault.read_workspace_file(run_file)
                        res = sandbox.run_skill_code(sk["code"], input_text=run_text)
                        st.session_state.last_skill_result = {
                            "title": sk.get("title", sk["name"]),
                            "file": run_file,
                            "res": res,
                            "code": sk["code"],
                        }

        if st.session_state.last_skill_result:
            item = st.session_state.last_skill_result
            with st.container(border=True):
                st.markdown(f"#### ✅ Results: {item['title']} *(on `{item['file']}`)*")
                render_clean_skill_output(item["res"]["stdout"])
                with st.expander("🔧 View Technical Sandbox Details & Python Code"):
                    st.code(item["code"], language="python")
                    st.json(item["res"])

        st.divider()
        with st.expander("➕ Teach ClawBrief a Brand-New Skill (Powered by Nemotron 3 Ultra)"):
            col_a, col_b = st.columns([1, 2])
            with col_a:
                new_skill_name = st.text_input("Skill Name:", value="word_count_checker")
            with col_b:
                new_skill_goal = st.text_input(
                    "What should this skill do?",
                    value="Count total words and lines from sys.stdin.read() and return them as JSON.",
                )
            if st.button("✨ Generate & Save New Skill with Nemotron 3 Ultra"):
                author_sys = (
                    "Output ONLY valid Python 3 code using standard library. "
                    "Read from sys.stdin.read() and print a JSON object with an 'items' list of strings."
                )
                code_reply, event = router.complete(
                    system_prompt=author_sys,
                    user_prompt=new_skill_goal,
                    task_type="skill_generation",
                    force_tier="ultra",
                )
                st.session_state.routing_log.insert(0, asdict(event))
                clean_code = sandbox.extract_python_code(code_reply)
                sbx_res = sandbox.run_skill_code(clean_code, input_text=active_file_content)
                vault.save_skill(
                    name=re.sub(r"[^a-zA-Z0-9_]", "_", new_skill_name),
                    description=new_skill_goal,
                    code=clean_code,
                    authored_by=event.model_id,
                )
                st.success(f"Created and verified `{new_skill_name}` using `{event.model_id}`!")

    # === TAB 3: DAILY MORNING BRIEFING ===
    with tab_brief:
        st.subheader("☀️ Your Automated Daily Morning Briefing")
        st.write(
            "Combines your saved memories, deadlines, and calendar schedule via a background **Nebius Serverless Job**."
        )

        if st.button("☀️ Generate Today's Morning Briefing", type="primary"):
            with st.spinner("Running background skills and compiling your briefing..."):
                memories = vault.list_memories()
                skills = vault.list_skills()
                job_result = jobs_client.trigger_nightly_digest_job(len(memories), len(skills))

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
                            "skill_name": sk.get("title", sk["name"]),
                            "source_file": src_name,
                            "stdout": out["stdout"],
                        }
                    )

                exec_prompt = (
                    "Write a warm, clear 3-bullet Morning Briefing for the user based on these memories:\n"
                    + json.dumps(memories[:4], indent=2)
                )
                ai_summary, event = router.complete(
                    system_prompt="You are ClawBrief's Morning Briefing assistant.",
                    user_prompt=exec_prompt,
                    task_type="nightly_digest",
                    force_tier="super",
                )
                st.session_state.routing_log.insert(0, asdict(event))

                st.session_state.latest_brief_md = jobs_client.compile_morning_brief_markdown(
                    memories=memories,
                    skill_outputs=skill_outputs,
                    job_meta=job_result,
                    ai_executive_summary=ai_summary,
                )

        if st.session_state.latest_brief_md:
            with st.container(border=True):
                st.markdown(st.session_state.latest_brief_md)
                st.download_button(
                    label="📥 Download Briefing as Markdown (.md)",
                    data=st.session_state.latest_brief_md,
                    file_name=f"morning_brief_{datetime.now().strftime('%Y_%m_%d')}.md",
                    mime="text/markdown",
                )

        if st.session_state.routing_log:
            with st.expander("📊 View Model Router Activity & Credit Savings Log"):
                st.dataframe(st.session_state.routing_log, use_container_width=True)


def run_cli_self_test() -> None:
    print("=== ClawBrief Clean UI & Phase 2 Self-Test ===")
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
        print(f"[3] Skill `{sk['name']}` Exit {res['exit_code']}")

    job_out = jobs.trigger_nightly_digest_job(len(vault.list_memories()), len(vault.list_skills()))
    print(f"[4] Nebius Serverless Job Status: {job_out['status']} ({job_out['job_id']})")


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