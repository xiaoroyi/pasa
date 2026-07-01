import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "evaluate_spar.py"
SPEC = importlib.util.spec_from_file_location("evaluate_spar", MODULE_PATH)
evaluate_spar = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluate_spar
SPEC.loader.exec_module(evaluate_spar)


class EvaluateSparTests(unittest.TestCase):
    def test_auto_scholar_ids_and_missing_result(self):
        rows = [
            {
                "question": "first query",
                "answer": ["A Paper"],
                "answer_arxiv_id": ["2401.01234v1"],
                "qid": "q1",
            },
            {
                "question": "second query",
                "answer": ["Missing Paper"],
                "answer_arxiv_id": ["2402.05678"],
                "qid": "q2",
            },
        ]
        results = {
            "first query": {
                "search_query": "first query",
                "docs": [],
                "children": [
                    {
                        "docs": [
                            {
                                "paper_id": "https://arxiv.org/abs/2401.01234v2",
                                "sim_score": 0.9,
                            }
                        ],
                        "children": [],
                    }
                ],
                "extra": {
                    "searched_docs": {
                        "2401.01234": {
                            "paper_id": "2401.01234",
                            "title": "A Paper",
                        }
                    }
                },
            }
        }

        report = evaluate_spar.evaluate(rows, results)
        macro = report["summary"]["macro"]
        micro = report["summary"]["micro"]

        self.assertEqual(report["summary"]["results_found"], 1)
        self.assertAlmostEqual(macro["precision"], 0.5)
        self.assertAlmostEqual(macro["recall"], 0.5)
        self.assertAlmostEqual(macro["f1"], 0.5)
        self.assertEqual((micro["tp"], micro["fp"], micro["fn"]), (1, 0, 1))

    def test_sparbench_title_fallback_and_false_positive(self):
        rows = [
            {
                "question": "query",
                "source_meta": {
                    "answers": [
                        {
                            "paperID": "https://doi.org/10.1000/example",
                            "title": "Paper 2: A Result",
                        }
                    ]
                },
            }
        ]
        results = {
            "query": {
                "search_query": "query",
                "docs": [
                    {"paper_id": "unknown-1", "title": "Paper 2 - A Result"},
                    {"paper_id": "unknown-2", "title": "Unrelated"},
                ],
                "children": [],
                "extra": {"searched_docs": {}},
            }
        }

        report = evaluate_spar.evaluate(rows, results)
        macro = report["summary"]["macro"]

        self.assertAlmostEqual(macro["precision"], 0.5)
        self.assertAlmostEqual(macro["recall"], 1.0)
        self.assertAlmostEqual(macro["f1"], 2 / 3)

    def test_raw_and_filtered_sets_are_separate(self):
        row = {
            "question": "query",
            "answer": ["Relevant"],
            "answer_arxiv_id": ["2401.00001"],
        }
        result = {
            "search_query": "query",
            "docs": [{"paper_id": "2401.00001"}],
            "children": [],
            "extra": {
                "searched_docs": {
                    "2401.00001": {
                        "paper_id": "2401.00001",
                        "title": "Relevant",
                        "sim_score": 0.9,
                    },
                    "2401.00002": {
                        "paper_id": "2401.00002",
                        "title": "Noise",
                        "sim_score": 0.1,
                    },
                }
            },
        }

        filtered = evaluate_spar.evaluate([row], {"query": result})
        raw = evaluate_spar.evaluate(
            [row], {"query": result}, prediction_set="raw"
        )

        self.assertAlmostEqual(filtered["summary"]["macro"]["precision"], 1.0)
        self.assertAlmostEqual(raw["summary"]["macro"]["precision"], 0.5)


if __name__ == "__main__":
    unittest.main()
