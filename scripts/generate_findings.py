"""Automated Findings Generation CLI Script.

Reads completed empirical results from the persistent research database and updates
the research_findings table.

Usage:
    python scripts/generate_findings.py [--run-id STR]
"""

import sys
import argparse
from pathlib import Path

# Add src to sys.path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from quant_engine.research.store import ResearchStore
from quant_engine.research.findings_generator import FindingsGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate and persist empirical research findings from database.")
    parser.add_argument("--run-id", type=str, default=None, help="Target research run ID (defaults to latest run).")
    args = parser.parse_args()

    store = ResearchStore()
    latest_run = store.get_latest_run()
    run_id = args.run_id or (latest_run.get("run_id") if latest_run else "RUN-MANUAL")

    print(f"=== GENERATING RESEARCH FINDINGS FOR RUN: {run_id} ===")
    findings_gen = FindingsGenerator(store)
    findings = findings_gen.generate_all_findings(run_id=run_id)

    if not findings:
        print("No strategy results found in database. Run 'python scripts/run_research.py' first.")
        return

    store.save_findings(findings)
    print(f"Successfully generated and saved {len(findings)} structured findings:")
    for f in findings:
        print(f"  [{f['severity']}] {f['headline']}")


if __name__ == "__main__":
    main()
