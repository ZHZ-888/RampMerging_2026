"""Aggregate per-scenario LIPR Dispatch audit results by model."""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def _number(row, name):
    return float(row.get(name, 0) or 0)


def _weighted_rate(rows, rate_name, count_name):
    denominator = sum(_number(row, count_name) for row in rows)
    if denominator == 0:
        return 0.0
    numerator = sum(
        _number(row, rate_name) * _number(row, count_name)
        for row in rows
    )
    return numerator / denominator


def summarise_model(rows):
    first = rows[0]
    ce_count = sum(_number(row, "ce_count") for row in rows)
    se_count = sum(_number(row, "se_count") for row in rows)
    overall_count = sum(_number(row, "overall_count") for row in rows)
    ce_gated_sum = sum(_number(row, "ce_gated_reward_sum") for row in rows)
    se_gated_sum = sum(_number(row, "se_gated_reward_sum") for row in rows)
    overall_gated_sum = sum(
        _number(row, "overall_gated_reward_sum") for row in rows
    )
    ce_score = ce_gated_sum / ce_count if ce_count else 0.0
    se_score = se_gated_sum / se_count if se_count else 0.0
    available_scores = [
        score
        for score, count in ((ce_score, ce_count), (se_score, se_count))
        if count > 0
    ]
    scenario_scores = [
        _number(row, "primary_macro_gated_reward") for row in rows
    ]
    decisions = sum(_number(row, "audit_decision_count") for row in rows)
    matched = sum(_number(row, "audit_matched_count") for row in rows)

    return {
        "model_id": int(float(first["model_id"])),
        "learning_rate": _number(first, "learning_rate"),
        "train_interval": int(_number(first, "train_interval")),
        "scenario_count": len(rows),
        "primary_macro_gated_reward": (
            sum(available_scores) / len(available_scores)
            if available_scores else 0.0
        ),
        "scenario_macro_mean": statistics.mean(scenario_scores),
        "scenario_macro_std": (
            statistics.pstdev(scenario_scores)
            if len(scenario_scores) > 1 else 0.0
        ),
        "ce_gated_reward_avg": ce_score,
        "se_gated_reward_avg": se_score,
        "overall_gated_reward_avg": (
            overall_gated_sum / overall_count if overall_count else 0.0
        ),
        "overall_execution_rate": _weighted_rate(
            rows, "overall_execution_rate", "overall_count"
        ),
        "overall_positive_retention_rate": _weighted_rate(
            rows, "overall_positive_retention_rate", "overall_positive_count"
        ),
        "overall_failure_rejection_rate": _weighted_rate(
            rows, "overall_failure_rejection_rate", "overall_nonpositive_count"
        ),
        "overall_decision_accuracy": _weighted_rate(
            rows, "overall_decision_accuracy", "overall_count"
        ),
        "audit_match_rate": matched / decisions if decisions else 0.0,
        "audit_unmatched_decision_count": int(sum(
            _number(row, "audit_unmatched_decision_count") for row in rows
        )),
        "audit_unmatched_reward_count": int(sum(
            _number(row, "audit_unmatched_reward_count") for row in rows
        )),
        "mean_SPR": statistics.mean(_number(row, "SPR") for row in rows),
        "total_runtime": sum(_number(row, "runtime") for row in rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    result_root = Path(args.result_root)
    result_files = sorted((result_root / "results").glob("model_*.csv"))
    if not result_files:
        raise FileNotFoundError(
            f"No model result CSV files found under {result_root / 'results'}"
        )

    grouped = defaultdict(list)
    for result_file in result_files:
        with result_file.open(newline="") as file:
            for row in csv.DictReader(file):
                grouped[int(float(row["model_id"]))].append(row)

    ranking = [summarise_model(rows) for rows in grouped.values()]
    ranking.sort(key=lambda row: row["primary_macro_gated_reward"], reverse=True)
    output_path = Path(args.out_csv) if args.out_csv else (
        result_root / "dispatch_model_ranking.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(ranking[0]))
        writer.writeheader()
        writer.writerows(ranking)

    best = ranking[0]
    print(f"Saved ranking: {output_path}")
    print(
        f"Best model: {best['model_id']} "
        f"(LR={best['learning_rate']}, I={best['train_interval']}, "
        f"macro gated reward={best['primary_macro_gated_reward']:.4f})"
    )


if __name__ == "__main__":
    main()
