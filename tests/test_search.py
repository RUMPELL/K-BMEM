import math
import unittest

from src.kbmem.search import cosine_ranking, l2_normalize
from src.kbmem.cli import build_parser


class RetrievalTests(unittest.TestCase):
    def test_normalization(self):
        self.assertAlmostEqual(sum(x * x for x in l2_normalize([3, 4])), 1.0)

    def test_cosine_ranking(self):
        ranked = cosine_ranking([1, 0], [("b", [0, 1]), ("a", [1, 0])], top_k=2)
        self.assertEqual([row[0] for row in ranked], ["a", "b"])

    def test_zero_vector_rejected(self):
        with self.assertRaises(ValueError):
            l2_normalize([0, 0])

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            l2_normalize([math.inf, 1])

    def test_cli_requires_model_and_query(self):
        args = build_parser().parse_args(["--model", "local-model", "--query", "synthetic query"])
        self.assertEqual(args.model, "local-model")
        self.assertEqual(args.query, "synthetic query")


if __name__ == "__main__":
    unittest.main()
