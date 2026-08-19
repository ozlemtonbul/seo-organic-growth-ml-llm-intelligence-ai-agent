from __future__ import annotations

from datetime import datetime
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = REPORT_DIR / "final_qa_report.txt"


def run_step(title: str, command: list[str]) -> tuple[int, str]:
    line = "=" * 72
    print(f"\n{line}\n{title}\n{line}")
    print(">", " ".join(command))

    process = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    output = (
        (process.stdout or "")
        + ("\n" if process.stdout and process.stderr else "")
        + (process.stderr or "")
    ).strip()

    print(output)
    return process.returncode, output


def main() -> int:
    sections = []
    overall_ok = True

    code, output = run_step(
        "1/4 PYTHON COMPILE",
        [sys.executable, "-m", "compileall", "dashboard", "src", "scripts"],
    )
    sections.append(("PYTHON COMPILE", code, output))
    overall_ok &= code == 0

    validation_script = PROJECT_ROOT / "scripts" / "validate_dashboard_data.py"
    if validation_script.exists():
        code, output = run_step(
            "2/4 PIPELINE DATA VALIDATION",
            [sys.executable, str(validation_script)],
        )
    else:
        code, output = 1, "scripts/validate_dashboard_data.py was not found."
        print(output)
    sections.append(("PIPELINE DATA VALIDATION", code, output))
    overall_ok &= code == 0

    code, output = run_step(
        "3/4 BUSINESS LOGIC + LOCALIZATION + UI AUTOMATION",
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/final_qa",
            "-q",
            "-ra",
        ],
    )
    sections.append(("FINAL PYTEST", code, output))
    overall_ok &= code == 0

    status = "PASS" if overall_ok else "FAIL"
    summary = [
        "",
        "=" * 72,
        f"FINAL QA STATUS: {status}",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "=" * 72,
    ]

    for title, code, _ in sections:
        summary.append(f"{title}: {'PASS' if code == 0 else 'FAIL'}")

    report_parts = [
        "\n".join(summary),
    ]

    for title, code, output in sections:
        report_parts.append(
            f"\n\n{'=' * 72}\n{title} | exit={code}\n{'=' * 72}\n{output}"
        )

    REPORT_PATH.write_text(
        "\n".join(report_parts),
        encoding="utf-8",
    )

    print("\n".join(summary))
    print(f"\nFull report: {REPORT_PATH}")

    if overall_ok:
        print(
            "\nAll automated checks passed. "
            "Only a short visual smoke check remains before demo/GitHub."
        )
        return 0

    print(
        "\nOne or more checks failed. "
        "Send the terminal output or outputs/final_qa_report.txt for targeted fixes."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
