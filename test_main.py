import math
import unittest

from main import LOWER, UPPER, objective, run_ga, variant


class Lab1Tests(unittest.TestCase):
    def test_variant_and_known_minimum(self):
        self.assertEqual(variant(23, 527, 26), 12)
        x = [(-1) ** i * math.sqrt(i) for i in range(1, 6)]
        self.assertAlmostEqual(objective(x), 0.0, places=12)
        self.assertEqual(objective([0.0] * 5), 55.0)

    def test_reproducibility_and_bounds(self):
        config = {
            "population_size": 30,
            "generations": 8,
            "tournament_size": 3,
            "crossover_probability": 0.9,
            "mutation_sigma": 5.0,
        }
        first = run_ga(config, 0.1, 2026, 5)
        second = run_ga(config, 0.1, 2026, 5)
        self.assertEqual(first, second)
        x, value, history = first
        self.assertTrue(all(LOWER <= coordinate <= UPPER for coordinate in x))
        self.assertAlmostEqual(objective(x), value)
        self.assertTrue(all(a >= b for a, b in zip(history, history[1:])))


if __name__ == "__main__":
    unittest.main()
