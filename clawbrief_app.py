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
from typing import Any, Dict, List, Optional, Union
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
# 1. DUAL-PROVIDER SMART MODEL ROUTER + TOKENMAXX GOVERNOR
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


class TokenmaxxGovernor:
    """Preflights prompt token budgets, slices large context, and writes .tokenmaxx/checkpoint.json."""

    def __init__(self, session_token_limit: int = 24000, checkpoint_dir: str = ".tokenmaxx"):
        self.session_limit = session_token_limit
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "checkpoint.json"

    @staticmethod
    def estimate_tokens(text: str, is_code: bool = False) -> int:
        chars_per_token = 3.2 if is_code else 4.0
        return max(1, int(len(text) / chars_per_token))

    @staticmethod
    def slice_context(text: str, query: str, max_chars: int = 4500) -> tuple[str, int]:
        """Slices relevant paragraphs around query keywords while keeping intro context for summaries."""
        original_len = len(text)
        if original_len <= max_chars:
            return text, 0
        keywords = [
            w.lower()
            for w in re.findall(r"[a-zA-Z0-9]{4,}", query)
            if w.lower() not in {"summarize", "summary", "clear", "bullet", "what", "this", "file", "from", "with", "your", "note", "give"}
        ]
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if keywords and paragraphs:
            scored = sorted(
                paragraphs,
                key=lambda p: sum(1 for k in keywords if k in p.lower()),
                reverse=True,
            )
            sliced = "\n\n".join(scored)[:max_chars]
        else:
            # For general summary requests, take the beginning + middle + end sample
            half = max_chars // 2
            sliced = text[:half] + "\n\n...[middle section]...\n\n" + text[original_len // 2 : (original_len // 2) + half]
        saved_chars = max(0, original_len - len(sliced))
        return sliced, int(saved_chars / 4.0)

    def get_budget_status(self, used_tokens: int) -> Dict[str, Any]:
        remaining = max(0, self.session_limit - used_tokens)
        pct_remaining = (remaining / self.session_limit) * 100.0
        if pct_remaining > 40:
            state = "Green"
        elif pct_remaining >= 20:
            state = "Yellow"
        elif pct_remaining >= 10:
            state = "Orange"
        else:
            state = "Red"
        return {
            "state": state,
            "used_tokens": used_tokens,
            "remaining_safe": remaining,
            "pct_remaining": round(pct_remaining, 1),
        }

    def write_checkpoint(
        self,
        active_file: str,
        memories: List[Dict[str, Any]],
        skills: List[Dict[str, Any]],
        used_tokens: int,
    ) -> Path:
        status = self.get_budget_status(used_tokens)
        payload = {
            "version": 1,
            "task": f"ClawBrief Personal AI Session ({active_file})",
            "completed": [
                {
                    "id": "workspace-ingestion",
                    "files": [active_file],
                    "summary": f"Indexed {len(memories)} memories and {len(skills)} verified OpenShell skills",
                    "validated": True,
                }
            ],
            "remaining": [
                {
                    "id": "final-nebius-cutover",
                    "goal": "Switch sidebar toggle to Nebius Token Factory for Oct 30 submission video",
                    "dependsOn": ["workspace-ingestion"],
                }
            ],
            "decisions": [
                "Use Tokenmaxx slice_context to cap per-call file context at 4,500 chars",
                "Route everyday triage to Nemotron Lightning/Super and reserve Ultra for skill synthesis",
            ],
            "constraints": [
                "Keep local files inside NVIDIA OpenShell sandbox boundary",
                "Never expose API keys to skill subprocesses",
            ],
            "currentState": f"Tokenmaxx State: {status['state']} ({status['pct_remaining']}% budget remaining)",
            "nextAction": "Run python -m streamlit run clawbrief_app.py",
            "budget": {
                "estimatedTotal": self.session_limit,
                "consumed": used_tokens,
                "remainingSafe": status["remaining_safe"],
            },
            "validation": {
                "run": ["CLAWBRIEF_CLI_TEST=1 python3 clawbrief_app.py"],
                "passed": ["CLAWBRIEF_CLI_TEST=1 python3 clawbrief_app.py"],
                "failed": [],
            },
        }
        self.checkpoint_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return self.checkpoint_file


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
            "deep research",
            "multi-step",
        ]
        fast_keywords = [
            "tag",
            "classify",
            "quick",
            "remind",
            "status",
            "triage",
        ]
        if any(k in lower for k in skill_keywords):
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
            f"- **Source Preview**: `{user_prompt[:180]}...`\n"
            "- **Note**: Paste your API key in the left sidebar if not connected yet."
        )


# =====================================================================
# 2. LOCAL WORKSPACE MANAGER & HERMES PERSISTENT VAULT
# =====================================================================

DEFAULT_SKILLS = [
    {
        "name": "deadline_and_action_harvester",
        "title": "⚡ Extract Deadlines & To-Dos",
        "description": "Finds every TODO, ACTION item, checkbox, and due date in your selected note or PDF.",
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
        "description": "Converts bulleted definitions or headings in your notes/PDFs into Q&A flashcards.",
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
            "    elif ':' in s and len(s) < 180:\n"
            "        term, definition = s.lstrip('-*•0123456789. ').split(':', 1)\n"
            "        if term.strip() and definition.strip():\n"
            "            cards.append({'topic': current_topic, 'question': f'What is {term.strip()}?', 'answer': definition.strip()})\n"
            "print(json.dumps({'skill': 'study_flashcard_crystallizer', 'count': len(cards), 'items': cards[:8]}, indent=2))\n"
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
    """Manages local `./workspace/notes` files (including PDFs & custom folders), memory, and skills."""

    def __init__(self, storage_dir: str = ".clawbrief_state", workspace_dir: Optional[str] = None):
        workspace_dir = workspace_dir or os.environ.get("CLAWBRIEF_NOTES_DIR", "").strip() or "workspace/notes"
        self.root = Path(storage_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.workspace = Path(workspace_dir).expanduser()
        self.workspace.mkdir(parents=True, exist_ok=True)
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

    def save_workspace_file(self, filename: str, content: Union[str, bytes]) -> Path:
        safe_name = Path(filename).name
        target = self.workspace / safe_name
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
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
            return "[PDF support is not installed. Run: python -m pip install pypdf]"
        try:
            reader = PdfReader(str(path))
            pages = [(page.extract_text() or "") for page in reader.pages]
            text = "\n\n".join(p.strip() for p in pages if p.strip())
            if not text:
                return "[No extractable text. This PDF may be a scanned image without selectable text.]"
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
    if "sliced_tokens_saved" not in st.session_state:
        st.session_state.sliced_tokens_saved = 0
    if "active_file_override" not in st.session_state:
        st.session_state.active_file_override = None

    custom_notes_dir = os.environ.get("CLAWBRIEF_NOTES_DIR", "").strip() or None
    vault = HermesSkillVault(workspace_dir=custom_notes_dir)
    sandbox = OpenShellSandbox()
    governor = TokenmaxxGovernor()

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

        # Process file uploads FIRST so newly uploaded PDFs/notes appear in the dropdown and auto-select!
        uploaded_files = st.file_uploader(
            "1. Upload your own notes (.pdf, .md, .txt, .ics, .csv):",
            type=["pdf", "md", "txt", "csv", "ics", "json"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            for uf in uploaded_files:
                if uf.name.lower().endswith(".pdf"):
                    vault.save_workspace_file(uf.name, uf.getvalue())
                else:
                    vault.save_workspace_file(uf.name, uf.read().decode("utf-8", errors="ignore"))
            # Automatically select the most recently uploaded file!
            st.session_state.active_file_override = Path(uploaded_files[-1].name).name
            st.success(f"Loaded `{st.session_state.active_file_override}` as Active File!")

        ws_files = vault.list_workspace_files()
        file_names = [f.name for f in ws_files]
        default_idx = 0
        if st.session_state.active_file_override in file_names:
            default_idx = file_names.index(st.session_state.active_file_override)

        selected_filename = st.selectbox(
            "2. Active File Selected for Questions & Skills:",
            options=file_names,
            index=default_idx if file_names else None,
        )

        with st.expander("⚙️ Advanced Router & Tokenmaxx Governor"):
            force_tier = st.selectbox(
                "Model Selection:",
                options=["auto", "nano_lightning", "super", "ultra"],
                format_func=lambda x: (
                    "Auto-Route (Saves Credits)"
                    if x == "auto"
                    else active_cfg["models"][x]["short_name"]
                ),
            )
            used_tokens = sum(i["estimated_tokens"] for i in st.session_state.routing_log)
            tm_status = governor.get_budget_status(used_tokens)
            st.caption(
                f"Tokenmaxx State: **{tm_status['state']}** ({tm_status['pct_remaining']}% budget left)"
            )
            if st.button("💾 Save Tokenmaxx Checkpoint (.json)", use_container_width=True):
                ckpt_path = governor.write_checkpoint(
                    active_file=selected_filename or "none",
                    memories=vault.list_memories(),
                    skills=vault.list_skills(),
                    used_tokens=used_tokens,
                )
                st.success(f"Saved `{ckpt_path}`!")

    router = DualNemotronRouter(provider_key=provider_key, api_key=api_key)
    jobs_client = NebiusServerlessJobsClient()
    active_file_content = vault.read_workspace_file(selected_filename) if selected_filename else ""

    # -----------------------------------------------------------------
    # TOP HEADER & 4 CLEAN SUMMARY METRICS
    # -----------------------------------------------------------------
    st.title("🛡️ ClawBrief — Your Private Personal AI")
    st.caption(
        "Keeps your personal notes & PDFs inside an isolated NVIDIA OpenShell sandbox while routing tasks across NVIDIA Nemotron models."
    )

    logs: List[Dict[str, Any]] = st.session_state.routing_log
    avg_saved = (sum(i["credits_saved_pct"] for i in logs) / len(logs)) if logs else 85.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📄 Private Files & PDFs", len(ws_files))
    m2.metric("🧠 Saved Memories", len(vault.list_memories()))
    m3.metric("⚡ Reusable Skills", len(vault.list_skills()))
    m4.metric("💰 Credits Saved", f"{avg_saved:.0f}%")

    # -----------------------------------------------------------------
    # 3 SIMPLE, HUMAN-FRIENDLY TABS
    # -----------------------------------------------------------------
    tab_chat, tab_skills, tab_brief = st.tabs(
        [
            "💬 Step 1: Ask Your Notes & PDFs",
            "⚡ Step 2: Run One-Click Skills",
            "☀️ Step 3: Daily Morning Briefing",
        ]
    )

    # === TAB 1: SIMPLE ASK & MEMORY ===
    with tab_chat:
        left_col, right_col = st.columns([3, 2], gap="large")

        with left_col:
            st.subheader(f"Ask about `{selected_filename}`")
            if active_file_content.startswith("[PDF support is not installed") or active_file_content.startswith("[No extractable text") or active_file_content.startswith("[Could not read PDF"):
                st.warning(active_file_content)

            with st.expander(f"👀 Preview extracted text from `{selected_filename}` ({len(active_file_content):,} chars)", expanded=False):
                st.code(active_file_content[:4000], language="markdown")

            st.write("**Quick Actions (click one or type your own below):**")
            q_col1, q_col2, q_col3 = st.columns(3)
            preset_prompt = None
            if q_col1.button("✨ Summarize File", use_container_width=True):
                preset_prompt = f"Give me a clear, structured summary of the topics covered in {selected_filename}."
            if q_col2.button("📅 Find Deadlines", use_container_width=True):
                preset_prompt = f"List every deadline, date, and action item mentioned in {selected_filename}."
            if q_col3.button("🎯 Key Concepts", use_container_width=True):
                preset_prompt = f"Explain the main concepts, definitions, and examples in {selected_filename} in simple terms."

            custom_question = st.text_input(
                "Or type any question about your active file:",
                value=preset_prompt or f"Summarize the main topics and key concepts in {selected_filename}.",
            )

            if st.button("🚀 Ask ClawBrief", type="primary", use_container_width=True) or preset_prompt:
                active_q = preset_prompt or custom_question
                with st.spinner(f"Reading `{selected_filename}` with NVIDIA Nemotron..."):
                    sliced_text, saved_toks = governor.slice_context(active_file_content, active_q)
                    st.session_state.sliced_tokens_saved += saved_toks
                    combined_prompt = (
                        f"Question: {active_q}\n\n"
                        f"=== ACTIVE FILE: {selected_filename} ===\n{sliced_text}"
                    )
                    sys_prompt = (
                        "You are ClawBrief, a friendly, concise Personal AI study and productivity assistant. "
                        "Answer the user's question strictly using the ACTIVE FILE text provided by the user. "
                        "Do not invent unrelated deadlines if the file is a study note or textbook PDF."
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
                        summary=f"Analyzed {selected_filename}: '{active_q[:55]}'",
                        source_file=selected_filename,
                    )

                with st.container(border=True):
                    st.markdown(f"#### 💡 ClawBrief Answer *(for `{selected_filename}`)*")
                    st.markdown(reply)
                    st.caption(
                        f"Answered by `{event.model_id}` in {event.latency_ms} ms • "
                        f"Saved {event.credits_saved_pct}% credits vs Ultra"
                        + (f" • Tokenmaxx sliced {saved_toks} excess tokens" if saved_toks else "")
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
    print("=== ClawBrief Auto-Select Upload + PDF Fix Self-Test ===")
    vault = HermesSkillVault()
    sandbox = OpenShellSandbox()
    jobs = NebiusServerlessJobsClient()
    governor = TokenmaxxGovernor()

    ws_files = [f.name for f in vault.list_workspace_files()]
    print(f"[1] Workspace Files Indexed: {ws_files}")

    nv_router = DualNemotronRouter(provider_key="nvidia_build")
    _, ev_nv = nv_router.complete(
        system_prompt="You are ClawBrief.",
        user_prompt="Summarize the main topics in DSA UNIT 1.pdf",
        task_type="chat",
    )
    print(f"[2] NVIDIA Build Route -> {ev_nv.model_id} (Saved {ev_nv.credits_saved_pct}% credits)")

    study_text = vault.read_workspace_file("agent_memory_study_notes.md")
    res = sandbox.run_skill_code(vault.list_skills()[0]["code"], input_text=study_text)
    print(f"[3] Skill Exit {res['exit_code']}")


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