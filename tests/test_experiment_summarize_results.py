import json
from pathlib import Path

from experiments.scripts.summarize_results import summarize_run


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_summarize_run_uses_top_candidate_per_request_for_primary_rates(tmp_path: Path) -> None:
    _write_text(
        tmp_path / "cases.jsonl",
        "\n".join(
            [
                json.dumps(
                    {
                        "request_id": "req-1",
                        "status": "FEASIBLE",
                        "reason_code": "OK",
                        "runtime_ms": 10,
                        "candidate_count": 2,
                    }
                ),
                json.dumps(
                    {
                        "request_id": "req-2",
                        "status": "FEASIBLE",
                        "reason_code": "OK",
                        "runtime_ms": 20,
                        "candidate_count": 1,
                    }
                ),
            ]
        )
        + "\n",
    )
    _write_text(
        tmp_path / "candidates.csv",
        "\n".join(
            [
                "engine_name,request_id,candidate_id,status,target_success,plausibility_pass,immutable_violation_count,mutable_violation_count,directional_violation_count,lof_score,distance_l1,changed_feature_count,delta",
                'dice,req-1,cf_1,FEASIBLE,True,True,0,0,0,1.0,0.1,1,"{}"',
                'dice,req-1,cf_2,FEASIBLE,False,False,1,0,0,3.0,0.5,2,"{}"',
                'dice,req-2,cf_1,FEASIBLE,True,True,0,0,0,1.2,0.2,1,"{}"',
            ]
        )
        + "\n",
    )

    summary = summarize_run(tmp_path)

    assert summary["requests_with_candidates"] == 2
    assert summary["target_success_rate"] == 1.0
    assert summary["plausibility_pass_rate"] == 1.0
    assert summary["immutable_violation_rate"] == 0.0
    assert summary["target_success_rate_all_candidates"] == (2 / 3)
    assert summary["plausibility_pass_rate_all_candidates"] == (2 / 3)
    assert summary["immutable_violation_rate_all_candidates"] == (1 / 3)
