import argparse
import csv
import re
import statistics

from collections import defaultdict
from pathlib import Path


# ============================================================
# Metadata parsing
# ============================================================


def find_first(
    text: str,
    candidates: list[str],
    default: str,
) -> str:

    for item in candidates:

        if item in text:
            return item

    return default


def parse_metadata(
    path: Path,
) -> dict:

    name = path.stem.lower()

    reward = find_first(
        name,
        [
            "shaped_v2",
            "shaped",
            "sparse",
        ],
        "unknown",
    )

    action_mode = find_first(
        name,
        [
            "pruned_shots",
            "full",
        ],
        "unknown",
    )

    layout_mode = find_first(
        name,
        [
            "random_medium",
            "random_easy",
            "fixed",
        ],
        "unknown",
    )

    # 注意顺序：
    # 长名字必须放在短名字前面。
    feature_mode = find_first(
        name,
        [
            "piece_transformer",
            "transformer",
            "coord_cnn",
            "unet_lite",
            "cnn_small",
            "cnn",
            "flat",
        ],
        "unknown",
    )

    geometry_mode = find_first(
        name,
        [
            "relative_v4",
            "relative_v3",
            "relative_v2",
            "relative_v1",
            "_none_",
        ],
        "none",
    )

    if geometry_mode == "_none_":
        geometry_mode = "none"

    seed_match = re.search(
        r"seed(\d+)",
        name,
    )

    seed = (
        int(seed_match.group(1))
        if seed_match
        else -1
    )

    config_id = (
        f"{reward}"
        f"_{action_mode}"
        f"_{layout_mode}"
        f"_{feature_mode}"
        f"_{geometry_mode}"
    )

    return {
        "reward": reward,
        "action_mode": action_mode,
        "layout_mode": layout_mode,
        "feature_mode": feature_mode,
        "geometry_mode": geometry_mode,
        "seed": seed,
        "config_id": config_id,

        # 用完整文件名作为唯一 run id
        "run_id": path.stem,
    }


# ============================================================
# Read all csv files
# ============================================================


def read_all_csv(
    input_dir: Path,
) -> list[dict]:

    rows = []

    csv_files = sorted(
        input_dir.glob("*.csv")
    )

    if not csv_files:

        raise RuntimeError(
            f"No CSV files found in: "
            f"{input_dir}"
        )

    for path in csv_files:

        meta = parse_metadata(
            path
        )

        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as f:

            reader = csv.DictReader(
                f
            )

            required = {
                "Wall time",
                "Step",
                "Value",
            }

            if not required.issubset(
                reader.fieldnames or []
            ):

                print(
                    "Skip invalid TensorBoard CSV:",
                    path.name,
                )

                continue

            for item in reader:

                row = {
                    **meta,

                    "wall_time": float(
                        item["Wall time"]
                    ),

                    "step": int(
                        float(
                            item["Step"]
                        )
                    ),

                    "win_rate": float(
                        item["Value"]
                    ),

                    "source_file":
                        path.name,
                }

                rows.append(
                    row
                )

    rows.sort(
        key=lambda x: (
            x["config_id"],
            x["seed"],
            x["step"],
        )
    )

    return rows


# ============================================================
# Long format
# ============================================================


def write_long(
    rows: list[dict],
    output_path: Path,
):

    fields = [
        "config_id",
        "run_id",

        "reward",
        "action_mode",
        "layout_mode",
        "feature_mode",
        "geometry_mode",

        "seed",

        "step",
        "win_rate",
        "wall_time",

        "source_file",
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

        writer.writerows(
            rows
        )


# ============================================================
# Wide format
# ============================================================


def write_wide(
    rows: list[dict],
    output_path: Path,
):

    steps = sorted(
        {
            row["step"]
            for row in rows
        }
    )

    run_ids = sorted(
        {
            row["run_id"]
            for row in rows
        }
    )

    value_map = {}

    for row in rows:

        value_map[
            (
                row["step"],
                row["run_id"],
            )
        ] = row["win_rate"]

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "Step",
                *run_ids,
            ]
        )

        for step in steps:

            line = [
                step
            ]

            for run_id in run_ids:

                value = value_map.get(
                    (
                        step,
                        run_id,
                    ),
                    "",
                )

                line.append(
                    value
                )

            writer.writerow(
                line
            )


# ============================================================
# Mean ± SD curve across seeds
# ============================================================


def write_seed_summary(
    rows: list[dict],
    output_path: Path,
):

    # 同一个 config / seed / step
    # 如果出现重复 CSV，
    # 只保留 wall_time 最新的一条。
    latest = {}

    for row in rows:

        key = (
            row["config_id"],
            row["seed"],
            row["step"],
        )

        old = latest.get(
            key
        )

        if (
            old is None
            or
            row["wall_time"]
            > old["wall_time"]
        ):

            latest[key] = row

    groups = defaultdict(
        list
    )

    for row in latest.values():

        key = (
            row["config_id"],
            row["step"],
        )

        groups[key].append(
            row["win_rate"]
        )

    result = []

    for (
        config_id,
        step,
    ), values in groups.items():

        mean = statistics.mean(
            values
        )

        if len(values) >= 2:

            std = statistics.stdev(
                values
            )

        else:

            std = 0.0

        result.append(
            {
                "config_id":
                    config_id,

                "step":
                    step,

                "mean_win_rate":
                    mean,

                "std_win_rate":
                    std,

                "n_seeds":
                    len(values),
            }
        )

    result.sort(
        key=lambda x: (
            x["config_id"],
            x["step"],
        )
    )

    fields = [
        "config_id",
        "step",
        "mean_win_rate",
        "std_win_rate",
        "n_seeds",
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

        writer.writerows(
            result
        )


# ============================================================
# Main
# ============================================================


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
        default="results/merged",
    )

    args = parser.parse_args()

    input_dir = Path(
        args.input_dir
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = read_all_csv(
        input_dir
    )

    long_path = (
        output_dir
        / "tensorboard_win_rate_long.csv"
    )

    wide_path = (
        output_dir
        / "tensorboard_win_rate_wide.csv"
    )

    summary_path = (
        output_dir
        / "tensorboard_win_rate_mean_std.csv"
    )

    write_long(
        rows,
        long_path,
    )

    write_wide(
        rows,
        wide_path,
    )

    write_seed_summary(
        rows,
        summary_path,
    )

    print(
        f"Loaded rows: {len(rows)}"
    )

    print(
        "Long CSV:",
        long_path,
    )

    print(
        "Wide CSV:",
        wide_path,
    )

    print(
        "Mean/Std CSV:",
        summary_path,
    )


if __name__ == "__main__":
    main()