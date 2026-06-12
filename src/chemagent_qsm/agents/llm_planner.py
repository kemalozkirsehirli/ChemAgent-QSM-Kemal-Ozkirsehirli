from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pydantic import ValidationError as PydanticValidationError

from chemagent_qsm.planner import HeuristicChemPlanner
from chemagent_qsm.schema import Backend, WorkflowPlan
from chemagent_qsm.validators import WorkflowValidator

SYSTEM_PROMPT = """
You are ChemAgent-QSM's planning module. Convert chemistry requests into strict JSON
compatible with chemagent_qsm.schema.WorkflowPlan. Do not emit Python code. Do not invent
files. Use PySCF-compatible QM tasks and statistical-mechanics task names only.
""".strip()


@dataclass
class PlanningAttempt:
    raw_text: str
    parsed: dict | None
    valid: bool
    errors: list[str]


class LLMPlanningAdapter:
    """Safe adapter that uses an LLM only to propose JSON plans, never to execute code.

    The callable interface is deliberately tiny: pass a function that takes a prompt string and
    returns text. That makes this adapter compatible with OpenAI, local models, LangChain,
    LangGraph, or hand-written test doubles without making any provider a hard dependency.
    """

    def __init__(self, llm: Callable[[str], str], validator: WorkflowValidator | None = None):
        self.llm = llm
        self.validator = validator or WorkflowValidator()
        self.fallback = HeuristicChemPlanner()

    def plan(self, prompt: str, backend: Backend | str = Backend.MOCK, output_dir: str | Path = "runs/llm_plan", max_repairs: int = 2) -> tuple[WorkflowPlan, list[PlanningAttempt]]:
        attempts: list[PlanningAttempt] = []
        request = self._build_request(prompt, backend, output_dir, feedback=None)
        for _ in range(max_repairs + 1):
            raw = self.llm(request)
            parsed = _extract_json(raw)
            if parsed is None:
                attempts.append(PlanningAttempt(raw, None, False, ["No JSON object found."]))
                request = self._build_request(prompt, backend, output_dir, feedback=attempts[-1].errors)
                continue
            try:
                plan = WorkflowPlan.model_validate(parsed)
            except PydanticValidationError as exc:
                errors = [e["msg"] for e in exc.errors()]
                attempts.append(PlanningAttempt(raw, parsed, False, errors))
                request = self._build_request(prompt, backend, output_dir, feedback=errors)
                continue
            static = self.validator.validate(plan)
            attempts.append(PlanningAttempt(raw, parsed, static["valid"], static["errors"] + static["warnings"]))
            if static["valid"]:
                return plan, attempts
            request = self._build_request(prompt, backend, output_dir, feedback=static["errors"])
        fallback = self.fallback.plan(prompt, backend=backend, output_dir=output_dir)
        fallback.metadata["llm_fallback"] = True
        fallback.metadata["llm_attempt_errors"] = [a.errors for a in attempts]
        return fallback, attempts

    def _build_request(self, prompt: str, backend: Backend | str, output_dir: str | Path, feedback: list[str] | None) -> str:
        schema_hint = {
            "backend": str(backend),
            "output_dir": str(output_dir),
            "qm_tasks": ["single_point", "optimize", "descriptors", "frequencies", "ir_spectrum"],
            "statmech_tasks": ["tcf", "msd", "sisf", "vacf", "rdf", "local_order", "relaxation", "mobility", "structure_dynamics_coupling", "non_gaussian", "mscope", "dynamical_heterogeneity"],
        }
        parts = [SYSTEM_PROMPT, "User request:", prompt, "Schema hints:", json.dumps(schema_hint, indent=2)]
        if feedback:
            parts.extend(["Validation feedback to repair:", json.dumps(feedback, indent=2)])
        parts.append("Return only one JSON object.")
        return "\n\n".join(parts)


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
