#!/usr/bin/env python3
"""Evaluate SPAR retrieval outputs on AutoScholarQuery or SPARBench."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlparse


ARXIV_RE = re.compile(
    r"(?:arxiv(?:\.org)?[:/](?:abs/|pdf/)?)?"
    r"(?P<id>\d{4}\.\d{4,5}|[a-z.-]+/\d{7})(?:v\d+)?",
    re.IGNORECASE,
)
DOI_RE = re.compile(r"10\.\d{4,9}/[^\s?#]+", re.IGNORECASE)
PMID_RE = re.compile(r"(?:pmid[:/\s]*|pubmed(?:\.ncbi\.nlm\.nih\.gov)?/)(\d+)", re.IGNORECASE)
OPENALEX_RE = re.compile(r"(?:openalex(?:\.org)?[:/])?(W\d+)", re.IGNORECASE)
S2_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)


@dataclass
class Paper:
    title: str = ""
    ids: set[str] = field(default_factory=set)

    @property
    def normalized_title(self) -> str:
        # This follows PASA's keep_letters behavior while supporting Unicode.
        return "".join(char.lower() for char in self.title if char.isalpha())

    @property
    def key(self) -> str:
        if self.ids:
            return sorted(self.ids)[0]
        if self.normalized_title:
            return f"title:{self.normalized_title}"
        return "unknown"


def normalize_identifier(value: Any) -> set[str]:
    if value is None:
        return set()
    text = unquote(str(value)).strip()
    if not text:
        return set()

    aliases: set[str] = set()
    lower = text.lower().strip()

    arxiv_match = ARXIV_RE.search(lower)
    if arxiv_match:
        aliases.add(f"arxiv:{arxiv_match.group('id').lower()}")

    doi_match = DOI_RE.search(lower)
    if doi_match:
        doi = doi_match.group(0).rstrip(".,);]")
        aliases.add(f"doi:{doi}")

    pmid_match = PMID_RE.search(lower)
    if pmid_match:
        aliases.add(f"pmid:{pmid_match.group(1)}")

    openalex_match = OPENALEX_RE.fullmatch(lower)
    if not openalex_match:
        openalex_match = re.search(r"openalex\.org/(W\d+)", text, re.IGNORECASE)
    if openalex_match:
        aliases.add(f"openalex:{openalex_match.group(1).lower()}")

    if S2_RE.fullmatch(lower):
        aliases.add(f"s2:{lower}")

    parsed = urlparse(text if "://" in text else "")
    if parsed.netloc:
        host = parsed.netloc.lower().removeprefix("www.")
        path = re.sub(r"\.pdf$", "", parsed.path.rstrip("/"), flags=re.IGNORECASE)
        aliases.add(f"url:{host}{path.lower()}")
    elif not any(char.isspace() for char in lower):
        aliases.add(f"id:{lower.removesuffix('.pdf').rstrip('/')}")

    return aliases


def paper_from_mapping(data: dict[str, Any]) -> Paper:
    title = str(data.get("title") or data.get("paper_title") or "")
    ids: set[str] = set()
    for field_name in (
        "paper_id",
        "paperID",
        "arxivId",
        "arxiv_id",
        "doi",
        "DOI",
        "pmid",
        "PMID",
        "id",
    ):
        ids.update(normalize_identifier(data.get(field_name)))
    return Paper(title=title, ids=ids)


def merge_duplicate_papers(papers: Iterable[Paper]) -> list[Paper]:
    merged: list[Paper] = []
    for paper in papers:
        if not paper.ids and not paper.normalized_title:
            continue
        duplicate = None
        for current in merged:
            same_id = bool(paper.ids and current.ids and paper.ids & current.ids)
            same_title = bool(
                paper.normalized_title
                and paper.normalized_title == current.normalized_title
            )
            if same_id or same_title:
                duplicate = current
                break
        if duplicate is None:
            merged.append(paper)
        else:
            duplicate.ids.update(paper.ids)
            if not duplicate.title and paper.title:
                duplicate.title = paper.title
    return merged


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def gold_papers(row: dict[str, Any]) -> list[Paper]:
    source_answers = (row.get("source_meta") or {}).get("answers") or []
    if source_answers:
        return merge_duplicate_papers(
            paper_from_mapping(answer)
            for answer in source_answers
            if isinstance(answer, dict)
        )

    titles = row.get("answer") or []
    identifiers = row.get("answer_arxiv_id") or []
    papers = []
    for index, title in enumerate(titles):
        paper_id = identifiers[index] if index < len(identifiers) else None
        papers.append(
            Paper(title=str(title or ""), ids=normalize_identifier(paper_id))
        )
    return merge_duplicate_papers(papers)


def iter_nodes(root: dict[str, Any]) -> Iterable[dict[str, Any]]:
    queue = [root]
    while queue:
        node = queue.pop(0)
        if not isinstance(node, dict):
            continue
        yield node
        children = node.get("children", [])
        if isinstance(children, list):
            queue.extend(child for child in children if isinstance(child, dict))


def searched_docs_by_id(root: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_docs = (root.get("extra") or {}).get("searched_docs") or {}
    lookup: dict[str, dict[str, Any]] = {}
    values: Iterable[Any]
    if isinstance(raw_docs, dict):
        values = raw_docs.values()
    elif isinstance(raw_docs, list):
        values = raw_docs
    else:
        values = []

    for doc in values:
        if not isinstance(doc, dict):
            continue
        for alias in paper_from_mapping(doc).ids:
            lookup[alias] = doc
    return lookup


def enrich_paper(doc: dict[str, Any], lookup: dict[str, dict[str, Any]]) -> Paper:
    paper = paper_from_mapping(doc)
    metadata = None
    for alias in paper.ids:
        if alias in lookup:
            metadata = lookup[alias]
            break
    if metadata:
        enriched = paper_from_mapping(metadata)
        paper.ids.update(enriched.ids)
        if not paper.title:
            paper.title = enriched.title
    return paper


def predicted_papers(
    root: dict[str, Any],
    prediction_set: str,
    score_threshold: float,
) -> list[Paper]:
    lookup = searched_docs_by_id(root)
    docs: list[dict[str, Any]] = []

    if prediction_set == "filtered":
        for node in iter_nodes(root):
            node_docs = node.get("docs") or []
            docs.extend(doc for doc in node_docs if isinstance(doc, dict))
    else:
        raw_docs = (root.get("extra") or {}).get("searched_docs") or {}
        if isinstance(raw_docs, dict):
            docs = [doc for doc in raw_docs.values() if isinstance(doc, dict)]
        elif isinstance(raw_docs, list):
            docs = [doc for doc in raw_docs if isinstance(doc, dict)]
        if prediction_set == "threshold":
            docs = [
                doc
                for doc in docs
                if float(doc.get("sim_score", -1) or -1) >= score_threshold
            ]

    return merge_duplicate_papers(enrich_paper(doc, lookup) for doc in docs)


def papers_match(prediction: Paper, gold: Paper, match_mode: str) -> bool:
    if match_mode in {"id", "id_or_title"} and prediction.ids & gold.ids:
        return True
    return bool(
        match_mode in {"title", "id_or_title"}
        and prediction.normalized_title
        and prediction.normalized_title == gold.normalized_title
    )


def maximum_matches(
    predictions: list[Paper],
    labels: list[Paper],
    match_mode: str,
) -> tuple[int, list[tuple[int, int]]]:
    adjacency = []
    for prediction in predictions:
        matches = [
            index
            for index, gold in enumerate(labels)
            if papers_match(prediction, gold, match_mode)
        ]
        matches.sort(
            key=lambda index: 0 if prediction.ids & labels[index].ids else 1
        )
        adjacency.append(matches)

    gold_to_prediction: dict[int, int] = {}

    def augment(prediction_index: int, visited: set[int]) -> bool:
        for gold_index in adjacency[prediction_index]:
            if gold_index in visited:
                continue
            visited.add(gold_index)
            previous = gold_to_prediction.get(gold_index)
            if previous is None or augment(previous, visited):
                gold_to_prediction[gold_index] = prediction_index
                return True
        return False

    for prediction_index in range(len(predictions)):
        augment(prediction_index, set())

    pairs = [
        (prediction_index, gold_index)
        for gold_index, prediction_index in gold_to_prediction.items()
    ]
    return len(pairs), sorted(pairs)


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def harmonic_mean(precision: float, recall: float) -> float:
    return safe_divide(2 * precision * recall, precision + recall)


def question_hash(question: str) -> str:
    return hashlib.md5(question.encode("utf-8")).hexdigest()


def load_result_index(results_folder: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    by_question: dict[str, dict[str, Any]] = {}
    errors = []
    for path in results_folder.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                result = json.load(handle)
            question = result.get("search_query")
            if question:
                by_question[str(question)] = result
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            errors.append(f"{path}: {exc}")
    return by_question, errors


def evaluate(
    benchmark_rows: list[dict[str, Any]],
    result_index: dict[str, dict[str, Any]],
    prediction_set: str = "filtered",
    match_mode: str = "id_or_title",
    score_threshold: float = 0.5,
) -> dict[str, Any]:
    details = []
    total_tp = total_fp = total_fn = 0
    precision_sum = recall_sum = 0.0
    found = 0

    for index, row in enumerate(benchmark_rows):
        question = str(row.get("question") or row.get("query") or "")
        result = result_index.get(question)
        labels = gold_papers(row)
        predictions = (
            predicted_papers(result, prediction_set, score_threshold)
            if result
            else []
        )
        found += int(result is not None)

        tp, pairs = maximum_matches(predictions, labels, match_mode)
        fp = len(predictions) - tp
        fn = len(labels) - tp
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        precision_sum += precision
        recall_sum += recall

        details.append(
            {
                "index": index,
                "qid": row.get("qid"),
                "question": question,
                "result_found": result is not None,
                "prediction_count": len(predictions),
                "gold_count": len(labels),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": harmonic_mean(precision, recall),
                "matches": [
                    {
                        "prediction": predictions[pred_index].title
                        or predictions[pred_index].key,
                        "gold": labels[gold_index].title or labels[gold_index].key,
                    }
                    for pred_index, gold_index in pairs
                ],
            }
        )

    query_count = len(benchmark_rows)
    macro_precision = safe_divide(precision_sum, query_count)
    macro_recall = safe_divide(recall_sum, query_count)
    micro_precision = safe_divide(total_tp, total_tp + total_fp)
    micro_recall = safe_divide(total_tp, total_tp + total_fn)

    return {
        "summary": {
            "query_count": query_count,
            "results_found": found,
            "results_missing": query_count - found,
            "prediction_set": prediction_set,
            "match_mode": match_mode,
            "macro": {
                "precision": macro_precision,
                "recall": macro_recall,
                "f1": harmonic_mean(macro_precision, macro_recall),
            },
            "micro": {
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
                "precision": micro_precision,
                "recall": micro_recall,
                "f1": harmonic_mean(micro_precision, micro_recall),
            },
        },
        "per_query": details,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate SPAR outputs. The default 'filtered' prediction set uses "
            "the docs retained by SPAR's relevance filter."
        )
    )
    parser.add_argument("--benchmark_file", type=Path, required=True)
    parser.add_argument("--results_folder", type=Path, required=True)
    parser.add_argument(
        "--prediction_set",
        choices=("filtered", "raw", "threshold"),
        default="filtered",
    )
    parser.add_argument(
        "--match_mode",
        choices=("id_or_title", "id", "title"),
        default="id_or_title",
    )
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Evaluate only the first N rows in the supplied benchmark file.",
    )
    parser.add_argument("--report_file", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.benchmark_file.is_file():
        print(f"Benchmark file not found: {args.benchmark_file}", file=sys.stderr)
        return 2
    if not args.results_folder.is_dir():
        print(f"Results folder not found: {args.results_folder}", file=sys.stderr)
        return 2

    rows = load_jsonl(args.benchmark_file)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        print("Benchmark contains no rows.", file=sys.stderr)
        return 2

    result_index, load_errors = load_result_index(args.results_folder)
    report = evaluate(
        rows,
        result_index,
        prediction_set=args.prediction_set,
        match_mode=args.match_mode,
        score_threshold=args.score_threshold,
    )
    report["metadata"] = {
        "benchmark_file": str(args.benchmark_file),
        "results_folder": str(args.results_folder),
        "load_errors": load_errors,
    }

    summary = report["summary"]
    macro = summary["macro"]
    micro = summary["micro"]
    print(
        f"Queries: {summary['query_count']} | "
        f"results found: {summary['results_found']} | "
        f"missing: {summary['results_missing']}"
    )
    print(
        "Paper-compatible macro: "
        f"F1={macro['f1']:.4f} "
        f"Recall={macro['recall']:.4f} "
        f"Precision={macro['precision']:.4f}"
    )
    print(
        "Micro: "
        f"F1={micro['f1']:.4f} "
        f"Recall={micro['recall']:.4f} "
        f"Precision={micro['precision']:.4f} "
        f"(TP={micro['tp']}, FP={micro['fp']}, FN={micro['fn']})"
    )
    if load_errors:
        print(f"Warning: failed to load {len(load_errors)} result file(s).")

    if args.report_file:
        args.report_file.parent.mkdir(parents=True, exist_ok=True)
        with args.report_file.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(f"Detailed report: {args.report_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
