#!/usr/bin/env python3
"""Contextual attribution analysis for Llama-shaped window profiles.

This script tests whether Single/Double/Triple measurements can predict Quad
windows by learning per-op contextual contributions:

  L(o1...on) ~= sum_i C(op_i | left_i, right_i)

The model is intentionally small and transparent. It uses additive features for
each token's op, left context, right context, and full local context, then fits
ridge regression on train lengths and evaluates held-out lengths.
"""
import argparse
import csv
import math
import os
from collections import Counter, defaultdict


HERE = os.path.dirname(os.path.abspath(__file__))
BOS = "BOS"
EOS = "EOS"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=os.path.join(HERE, "llama_context_windows_s128_results.csv"),
    )
    parser.add_argument(
        "--pred-out",
        default=os.path.join(HERE, "llama_context_attribution_predictions.csv"),
    )
    parser.add_argument(
        "--coef-out",
        default=os.path.join(HERE, "llama_context_attribution_coefficients.csv"),
    )
    parser.add_argument("--train-max-len", type=int, default=3)
    parser.add_argument("--test-len", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--families", nargs="*", default=None)
    return parser.parse_args()


def read_rows(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("error"):
                continue
            try:
                length = int(row["length"])
            except (TypeError, ValueError):
                continue
            row["length_int"] = length
            row["task_us_float"] = float(row["task_us"])
            rows.append(row)
    return rows


def token_contexts(sequence):
    for idx, op in enumerate(sequence):
        left = sequence[idx - 1] if idx > 0 else BOS
        right = sequence[idx + 1] if idx + 1 < len(sequence) else EOS
        yield idx, op, left, right


def featurize(sequence):
    features = Counter()
    for _, op, left, right in token_contexts(sequence):
        features[f"op:{op}"] += 1.0
        features[f"left:{left}->{op}"] += 1.0
        features[f"right:{op}->{right}"] += 1.0
        features[f"ctx:{left}|{op}|{right}"] += 1.0
    return features


def fit_ridge(samples, ridge):
    feature_names = sorted({name for features, _ in samples for name in features})
    index = {name: i for i, name in enumerate(feature_names)}
    n = len(feature_names)
    ata = [[0.0] * n for _ in range(n)]
    aty = [0.0] * n

    for features, target in samples:
        items = [(index[name], value) for name, value in features.items()]
        for i, vi in items:
            aty[i] += vi * target
            for j, vj in items:
                ata[i][j] += vi * vj

    for i in range(n):
        ata[i][i] += ridge

    coef = solve_linear_system(ata, aty)
    return feature_names, coef


def solve_linear_system(a, b):
    n = len(b)
    aug = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            continue
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        div = aug[col][col]
        for k in range(col, n + 1):
            aug[col][k] /= div
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor == 0:
                continue
            for k in range(col, n + 1):
                aug[r][k] -= factor * aug[col][k]
    return [aug[i][n] for i in range(n)]


def predict(features, feature_names, coef):
    coef_map = dict(zip(feature_names, coef))
    return sum(coef_map.get(name, 0.0) * value for name, value in features.items())


def contribution_rows(family, sequence, feature_names, coef):
    coef_map = dict(zip(feature_names, coef))
    rows = []
    for pos, op, left, right in token_contexts(sequence):
        parts = {
            "op": coef_map.get(f"op:{op}", 0.0),
            "left": coef_map.get(f"left:{left}->{op}", 0.0),
            "right": coef_map.get(f"right:{op}->{right}", 0.0),
            "ctx": coef_map.get(f"ctx:{left}|{op}|{right}", 0.0),
        }
        rows.append({
            "family": family,
            "sequence": sequence,
            "position": pos,
            "op": op,
            "left": left,
            "right": right,
            "op_coef": parts["op"],
            "left_coef": parts["left"],
            "right_coef": parts["right"],
            "ctx_coef": parts["ctx"],
            "token_contribution": sum(parts.values()),
        })
    return rows


def summarize_errors(pred_rows):
    by_family = defaultdict(list)
    for row in pred_rows:
        by_family[row["family"]].append(float(row["error_us"]))
    lines = []
    for family, errors in sorted(by_family.items()):
        abs_errors = sorted(abs(e) for e in errors)
        mean_abs = sum(abs_errors) / len(abs_errors)
        median_abs = abs_errors[len(abs_errors) // 2]
        max_abs = max(abs_errors)
        bias = sorted(errors)[len(errors) // 2]
        lines.append((family, len(errors), bias, median_abs, mean_abs, max_abs))
    return lines


def main():
    args = parse_args()
    rows = read_rows(args.input)
    if args.families:
        rows = [row for row in rows if row["family"] in set(args.families)]
    families = sorted({row["family"] for row in rows})

    all_pred_rows = []
    all_coef_rows = []
    for family in families:
        family_rows = [row for row in rows if row["family"] == family]
        train_rows = [row for row in family_rows if row["length_int"] <= args.train_max_len]
        test_rows = [row for row in family_rows if row["length_int"] == args.test_len]
        if not train_rows or not test_rows:
            continue

        samples = [
            (featurize(row["sequence"]), row["task_us_float"])
            for row in train_rows
        ]
        feature_names, coef = fit_ridge(samples, args.ridge)
        train_rmse = math.sqrt(
            sum(
                (
                    predict(featurize(row["sequence"]), feature_names, coef)
                    - row["task_us_float"]
                ) ** 2
                for row in train_rows
            ) / len(train_rows)
        )

        for row in test_rows:
            pred = predict(featurize(row["sequence"]), feature_names, coef)
            actual = row["task_us_float"]
            all_pred_rows.append({
                "family": family,
                "sequence": row["sequence"],
                "ops": row["ops"],
                "actual_us": actual,
                "pred_us": pred,
                "error_us": pred - actual,
                "abs_error_us": abs(pred - actual),
                "train_rows": len(train_rows),
                "features": len(feature_names),
                "ridge": args.ridge,
                "train_rmse_us": train_rmse,
            })
            all_coef_rows.extend(contribution_rows(family, row["sequence"], feature_names, coef))

    pred_fields = [
        "family", "sequence", "ops", "actual_us", "pred_us", "error_us",
        "abs_error_us", "train_rows", "features", "ridge", "train_rmse_us",
    ]
    with open(args.pred_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pred_fields)
        writer.writeheader()
        writer.writerows(all_pred_rows)

    coef_fields = [
        "family", "sequence", "position", "op", "left", "right",
        "op_coef", "left_coef", "right_coef", "ctx_coef", "token_contribution",
    ]
    with open(args.coef_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=coef_fields)
        writer.writeheader()
        writer.writerows(all_coef_rows)

    print(
        f"[DONE] predictions={args.pred_out} coefficients={args.coef_out}",
        flush=True,
    )
    print(
        f"train_len<= {args.train_max_len}, test_len={args.test_len}, ridge={args.ridge}",
        flush=True,
    )
    for family, n, bias, median_abs, mean_abs, max_abs in summarize_errors(all_pred_rows):
        print(
            f"{family:9s} n={n} bias={bias:+.2f}us "
            f"median_abs={median_abs:.2f}us mean_abs={mean_abs:.2f}us "
            f"max_abs={max_abs:.2f}us",
            flush=True,
        )


if __name__ == "__main__":
    main()
