"""Run all three capstone modules sequentially."""
import subprocess, sys

commands = [
    [sys.executable, "data_pipeline/cleaning.py"],
    [sys.executable, "data_pipeline/database.py"],
    [sys.executable, "data_pipeline/queries.py"],
    [sys.executable, "analytics/run_analytics.py"],
    [sys.executable, "-m", "support_assistant.main"],
]
for command in commands:
    print("\n>>>", " ".join(command))
    subprocess.run(command, check=True)
