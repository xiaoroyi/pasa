#!/usr/bin/env python3
"""Run SPAR over a benchmark and save one search tree per query."""

import argparse
import glob
import json
import os
import random
import shutil
import traceback
from datetime import datetime, timedelta
from pathlib import Path

from tqdm import tqdm

from global_config import (
    DO_FUSION_JUDGE,
    DO_REFERENCE_SEARCH,
    FUSION_TEMPLATE,
    SEARCH_ROUTES,
)
from pipeline_spar import AcademicSearchTree
from utils import get_md5


BENCHMARKS = {
    "AutoScholarQuery": "./benchmark/AutoScholarQuery_test.jsonl",
    "SPARBench": "./benchmark/spar_bench.jsonl",
    # Backward-compatible name used by the original README.
    "OwnBenchmark": "./benchmark/spar_bench.jsonl",
}

SOURCE_FILES = [
    "./global_config.py",
    "./instruction.py",
    "./run_spr_agent.py",
    "./search_engine.py",
    "./api_web.py",
    "./pipeline_spar.py",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=BENCHMARKS)
    parser.add_argument("--sample_num", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--score_thresh", type=float, default=0.5)
    parser.add_argument("--max_depth", type=int, default=2)
    parser.add_argument("--max_docs", type=int, default=10)
    parser.add_argument("--output_folder")
    parser.add_argument(
        "--use_end_date",
        action="store_true",
        help="Apply the benchmark publication date minus seven days.",
    )
    parser.add_argument("--skip_visualization", action="store_true")
    return parser.parse_args()


def default_output_folder(args):
    route_name = "-".join(SEARCH_ROUTES)
    return (
        f"./gen_result/{args.benchmark}_{args.sample_num}"
        f"_msearch_{route_name}"
        f"_depth{args.max_depth}"
        f"_do_reference_{DO_REFERENCE_SEARCH}"
        f"_query_judge_{DO_FUSION_JUDGE}"
        f"_fusion_{FUSION_TEMPLATE}"
        f"_enddate_{args.use_end_date}"
        f"_score_{args.score_thresh}"
    )


def load_benchmark(path, sample_num, seed):
    with open(path, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    random.Random(seed).shuffle(rows)
    return rows[:sample_num] if sample_num > 0 else rows


def result_index(output_folder):
    already = {}
    for path in glob.glob(os.path.join(output_folder, "*.json")):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                result = json.load(handle)
            question = result.get("search_query")
            if question:
                already[question] = path
        except (OSError, json.JSONDecodeError, AttributeError):
            traceback.print_exc()
    return already


def benchmark_end_date(row, enabled):
    if not enabled:
        return ""
    published_time = (row.get("source_meta") or {}).get("published_time")
    if not published_time:
        return ""
    return (
        datetime.strptime(str(published_time), "%Y%m%d") - timedelta(days=7)
    ).strftime("%Y%m%d")


def main():
    args = parse_args()
    src_file = BENCHMARKS[args.benchmark]
    output_folder = args.output_folder or default_output_folder(args)
    os.makedirs(output_folder, exist_ok=True)

    selected_file = os.path.join(
        "./benchmark",
        f"{Path(src_file).stem}_select_{args.sample_num}_seed{args.seed}.jsonl",
    )
    rows = load_benchmark(src_file, args.sample_num, args.seed)
    with open(selected_file, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    for source_file in SOURCE_FILES:
        shutil.copy2(source_file, output_folder)

    print(f"benchmark rows: {len(rows)}")
    print(f"selected benchmark: {selected_file}")
    print(f"output folder: {output_folder}")

    search_agent = AcademicSearchTree(
        max_depth=args.max_depth,
        max_docs=args.max_docs,
        similarity_threshold=args.score_thresh,
    )
    already = result_index(output_folder)

    for row in tqdm(rows, desc="Processing queries"):
        question = str(row.get("question") or row.get("query") or "")
        if not question:
            print("Skipping row without a question")
            continue
        if question in already:
            print(f"pass: {already[question]}")
            continue

        destination = os.path.join(output_folder, f"{get_md5(question)}.json")
        try:
            search_agent.search(
                question,
                end_date=benchmark_end_date(row, args.use_end_date),
            )
            if "answer" in row:
                search_agent.root.extra["answer"] = row["answer"]
            elif (row.get("source_meta") or {}).get("answers"):
                search_agent.root.extra["answer"] = [
                    answer.get("title", "")
                    for answer in row["source_meta"]["answers"]
                ]

            with open(destination, "w", encoding="utf-8") as handle:
                json.dump(
                    search_agent.root.convert_to_dict(),
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )

            if not args.skip_visualization:
                try:
                    search_agent.visualize_tree(os.path.splitext(destination)[0])
                except Exception:
                    traceback.print_exc()
        except Exception:
            traceback.print_exc()

    print(f"output folder: {output_folder}")
    print(
        "evaluate with: python evaluate_spar.py "
        f"--benchmark_file {selected_file} "
        f"--results_folder {output_folder}"
    )


if __name__ == "__main__":
    main()
