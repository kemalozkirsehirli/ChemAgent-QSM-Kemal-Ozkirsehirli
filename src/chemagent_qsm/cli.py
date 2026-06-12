from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich import print

from chemagent_qsm.orchestrator import ChemAgentQSM
from chemagent_qsm.planner import HeuristicChemPlanner, load_plan_from_yaml, save_plan_yaml
from chemagent_qsm.schema import Backend
from chemagent_qsm.validators import WorkflowValidator
from chemagent_qsm.evaluation.cv_objectives import evaluate_cv_objective_coverage
from chemagent_qsm.evaluation.benchmarks import evaluate_against_baselines

app = typer.Typer(help="ChemAgent-QSM: auditable agentic quantum/statistical-mechanics workflows.")


@app.command()
def plan(
    prompt: str = typer.Argument(..., help="Natural-language chemistry request."),
    backend: Backend = typer.Option(Backend.MOCK, help="Execution backend."),
    out: Path = typer.Option(Path("runs/planned"), help="Output directory encoded in the plan."),
    save: Optional[Path] = typer.Option(None, help="YAML path to save the plan."),
):
    workflow = HeuristicChemPlanner().plan(prompt, backend=backend, output_dir=out)
    if save:
        save_plan_yaml(workflow, save)
        print(f"[green]Saved plan:[/green] {save}")
    print(workflow.model_dump(mode="json"))


@app.command()
def validate(config: Path = typer.Option(..., help="Workflow YAML file.")):
    workflow = load_plan_from_yaml(config)
    report = WorkflowValidator().validate(workflow)
    print(report)
    raise typer.Exit(0 if report["valid"] else 1)


@app.command()
def run(
    config: Path = typer.Option(..., help="Workflow YAML file."),
    backend: Optional[Backend] = typer.Option(None, help="Override backend."),
    out: Optional[Path] = typer.Option(None, help="Override output directory."),
):
    workflow = load_plan_from_yaml(config)
    updates = {}
    if backend is not None:
        updates["backend"] = backend
    if out is not None:
        updates["output_dir"] = out
    if updates:
        workflow = workflow.model_copy(update=updates)
    result = ChemAgentQSM().run(workflow)
    print(f"[bold green]Workflow complete:[/bold green] {result['output_dir']}")


@app.command()
def demo(out: Path = typer.Option(Path("runs/demo_water"), help="Output directory.")):
    workflow = HeuristicChemPlanner().plan(
        "Optimize water with B3LYP/6-31g*, compute electronic descriptors and IR spectrum",
        backend=Backend.MOCK,
        output_dir=out,
    )
    result = ChemAgentQSM().run(workflow)
    print(f"[bold green]Demo complete:[/bold green] {result['output_dir']}")


@app.command("cv-check")
def cv_check(out: Path = typer.Option(..., help="Completed ChemAgent-QSM output directory to audit against the CV objective matrix.")):
    report = evaluate_cv_objective_coverage(out)
    print(report)
    raise typer.Exit(0 if report["passed"] else 1)


@app.command("benchmark-check")
def benchmark_check(out: Path = typer.Option(..., help="Completed ChemAgent-QSM output directory to benchmark against implementation baselines.")):
    report = evaluate_against_baselines(out)
    print(report)
    raise typer.Exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    app()
