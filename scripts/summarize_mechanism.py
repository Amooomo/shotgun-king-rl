"""Summarize per-seed mechanism JSON reports into machine-readable tables.

Input: JSON files produced by ``scripts/analyze_agent.py --output-json``.
Files are expected to be named like ``relative_v4_seed1_eval10000.json``.

Outputs (in ``--output-dir``):

* ``seed_metrics.csv``  -> one row per (geometry, seed)
* ``summary.csv``       -> Mean / sample SD / Min / Max per geometry
* ``matched_delta.csv`` -> per-seed and paired-mean delta (v4 - v2)
* ``summary.json``      -> same data, nested

Statistics are computed across training seeds (the unit of replication),
never across episodes.
"""

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


METRIC_NAMES = [
    "win_rate",
    "death_rate",
    "timeout_rate",
    "mean_return",
    "mean_length",
    "dangerous_state_exposure_rate",
    "dangerous_move_share",
    "dangerous_selection_rate",
    "dangerous_rate_given_move_in_danger_state",
    "overall_dangerous_action_rate",
]


def geometry_label(mode: str) -> str:

    if mode == "relative_v4":
        return "v4"

    if mode == "relative_v2":
        return "v2"

    return mode or "unknown"


def extract_metrics(
    report: dict,
) -> dict:

    episodes = int(report["episodes"])

    outcomes = report["outcomes"]

    episode = report["episode"]

    mechanism = report["mechanism"]

    return {
        "win_rate":
            outcomes["win"] / episodes,

        "death_rate":
            outcomes["death"] / episodes,

        "timeout_rate":
            outcomes["timeout"] / episodes,

        "mean_return":
            float(episode["mean_return"]),

        "mean_length":
            float(episode["mean_length"]),

        "dangerous_state_exposure_rate":
            float(
                mechanism[
                    "dangerous_state_exposure_rate"
                ]
            ),

        "dangerous_move_share":
            float(
                mechanism[
                    "dangerous_move_share"
                ]
            ),

        "dangerous_selection_rate":
            float(
                mechanism[
                    "dangerous_selection_rate"
                ]
            ),

        "dangerous_rate_given_move_in_danger_state":
            float(
                mechanism[
                    "dangerous_rate_given_move_in_danger_state"
                ]
            ),

        "overall_dangerous_action_rate":
            float(
                mechanism[
                    "overall_dangerous_action_rate"
                ]
            ),
    }


def load_reports(
    input_dir: Path,
) -> dict:

    data = defaultdict(dict)

    for path in sorted(
        input_dir.glob("*.json")
    ):

        match = re.search(
            r"seed(\d+)",
            path.stem,
        )

        if match is None:
            continue

        seed = int(match.group(1))

        with path.open(
            "r",
            encoding="utf-8",
        ) as f:

            report = json.load(f)

        label = geometry_label(
            report.get(
                "geometry_mode",
                "",
            )
        )

        data[label][seed] = extract_metrics(
            report
        )

    return data


def write_seed_metrics(
    data: dict,
    output_path: Path,
) -> None:

    fields = [
        "geometry",
        "seed",
        *METRIC_NAMES,
    ]

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for label in sorted(data):

            for seed in sorted(
                data[label]
            ):

                row = {
                    "geometry": label,
                    "seed": seed,
                    **data[label][seed],
                }

                writer.writerow(row)


def summarize(
    values: list[float],
) -> dict:

    n = len(values)

    if n == 0:

        return {
            "mean": 0.0,
            "sd": 0.0,
            "min": 0.0,
            "max": 0.0,
            "n": 0,
        }

    return {
        "mean":
            statistics.mean(values),

        "sd":
            statistics.stdev(values)
            if n >= 2
            else 0.0,

        "min":
            min(values),

        "max":
            max(values),

        "n":
            n,
    }


def write_summary(
    data: dict,
    output_path: Path,
) -> None:

    fields = [
        "geometry",
        "metric",
        "mean",
        "sd",
        "min",
        "max",
        "n",
    ]

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for label in sorted(data):

            for metric in METRIC_NAMES:

                values = [
                    data[label][seed][metric]
                    for seed in sorted(
                        data[label]
                    )
                ]

                stats = summarize(
                    values
                )

                writer.writerow(
                    {
                        "geometry": label,
                        "metric": metric,
                        **stats,
                    }
                )


def write_matched_delta(
    data: dict,
    output_path: Path,
) -> dict:

    matched = sorted(
        set(data.get("v4", {}))
        & set(data.get("v2", {}))
    )

    fields = [
        "metric",
        "seed",
        "delta",
    ]

    per_metric = defaultdict(list)

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for metric in METRIC_NAMES:

            for seed in matched:

                delta = (
                    data["v4"][seed][metric]
                    - data["v2"][seed][metric]
                )

                per_metric[metric].append(
                    delta
                )

                writer.writerow(
                    {
                        "metric": metric,
                        "seed": seed,
                        "delta": delta,
                    }
                )

            if per_metric[metric]:

                writer.writerow(
                    {
                        "metric": metric,
                        "seed":
                            "mean_paired",
                        "delta":
                            statistics.mean(
                                per_metric[
                                    metric
                                ]
                            ),
                    }
                )

    return {
        "matched_seeds": matched,
        "mean_paired_delta": {
            metric: (
                statistics.mean(values)
                if values
                else 0.0
            )
            for metric, values in (
                per_metric.items()
            )
        },
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else input_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_reports(input_dir)

    write_seed_metrics(
        data,
        output_dir / "seed_metrics.csv",
    )

    write_summary(
        data,
        output_dir / "summary.csv",
    )

    matched = write_matched_delta(
        data,
        output_dir / "matched_delta.csv",
    )

    with (
        output_dir / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "available_seeds": {
                    label: sorted(
                        seeds
                    )
                    for label, seeds in (
                        data.items()
                    )
                },
                **matched,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "Geometry / seeds:",
        {
            label: sorted(seeds)
            for label, seeds in data.items()
        },
    )

    print(
        "Matched seeds (v4 & v2):",
        matched["matched_seeds"],
    )

    print(
        "Wrote:",
        output_dir / "seed_metrics.csv",
    )

    print(
        "Wrote:",
        output_dir / "summary.csv",
    )

    print(
        "Wrote:",
        output_dir / "matched_delta.csv",
    )


if __name__ == "__main__":
    main()
