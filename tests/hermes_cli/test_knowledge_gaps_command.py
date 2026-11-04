"""The public Hermes CLI reaches the source-neutral proposal engine."""

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "tests" / "fixtures" / "knowledge_gaps"


def invoke(tmp_path, *args):
    env = os.environ.copy()
    env["HERMES_HOME"] = str(tmp_path / "hermes-home")
    # Execute the actual entry point and check that ordinary analysis never
    # loads the optional numeric experiment dependency.
    launcher = (
        "import runpy,sys; sys.argv=['hermes',*sys.argv[1:]]; "
        "runpy.run_module('hermes_cli.main',run_name='__main__'); "
        "assert 'numpy' not in sys.modules"
    )
    return subprocess.run(
        [sys.executable, "-c", launcher, "knowledge-gaps", *args],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=30,
    )


def test_public_cli_analyze_review_and_diff_use_the_real_engine(tmp_path):
    source = EXAMPLES / "input.json"
    proposals = EXAMPLES / "proposals.json"
    validated = invoke(tmp_path, "validate", "--input", str(source))
    assert validated.returncode == 0, validated.stderr
    assert json.loads(validated.stdout)["sources"] == len(json.loads(source.read_text())["sources"])
    first = invoke(tmp_path, "analyze", "--input", str(source))
    assert first.returncode == 0, first.stderr
    result = json.loads(first.stdout)
    assert result["proposal_only"] is True
    before = tmp_path / "before.json"
    before.write_text(first.stdout)
    second = invoke(tmp_path, "analyze", "--input", str(source))
    assert second.returncode == 0, second.stderr
    after = tmp_path / "after.json"
    after.write_text(second.stdout)
    diff = invoke(tmp_path, "diff", "--before", str(before), "--after", str(after))
    assert diff.returncode == 0, diff.stderr
    assert json.loads(diff.stdout)["changes"] == []
    review = invoke(tmp_path, "review", "--input", str(source), "--proposals", str(proposals))
    assert review.returncode == 0, review.stderr
    reviewed = json.loads(review.stdout)
    assert reviewed["validation"] == "unvalidated"
    assert reviewed["queue"]


def test_public_cli_refuses_hallucinated_evidence_without_publishing(tmp_path):
    proposed = json.loads((EXAMPLES / "proposals.json").read_text())
    proposed[0]["evidence"][0]["quote"] = "unrelated private source text"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(proposed))
    output = tmp_path / "review" / "result.json"
    result = invoke(tmp_path, "review", "--input", str(EXAMPLES / "input.json"),
                    "--proposals", str(bad), "--output", str(output))
    assert result.returncode == 2
    assert "unrelated private" not in result.stderr + result.stdout
    assert not output.exists()
