from __future__ import annotations

import unittest

from model_radar.fingerprint import HashFingerprint, fingerprint_repetitions
from model_radar.utils import cosine_similarity


class FingerprintTests(unittest.TestCase):
    def test_hash_fingerprint_is_stable_and_order_sensitive(self) -> None:
        backend = HashFingerprint(dimension=256, seed=7)
        left = backend.fingerprint(["alpha beta", "gamma delta"])
        again = backend.fingerprint(["alpha beta", "gamma delta"])
        swapped = backend.fingerprint(["gamma delta", "alpha beta"])
        self.assertAlmostEqual(cosine_similarity(left, again), 1.0, places=9)
        self.assertLess(cosine_similarity(left, swapped), 0.95)

    def test_repetition_aggregation_reports_noise(self) -> None:
        items = [
            {"repetition": 0, "prompt_index": 0, "response": "same answer"},
            {"repetition": 1, "prompt_index": 0, "response": "different answer"},
        ]
        vector, within, vectors = fingerprint_repetitions(
            items,
            HashFingerprint(dimension=64),
            prompt_count=1,
            repetitions=2,
        )
        self.assertEqual(len(vector), 64)
        self.assertEqual(len(vectors), 2)
        self.assertGreater(within, 0)


if __name__ == "__main__":
    unittest.main()

