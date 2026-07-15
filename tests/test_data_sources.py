import unittest
from stock_price_prediction.data_sources import moving_average, percent_change, normal_cdf, clamp


class TestMovingAverage(unittest.TestCase):
    def test_basic(self):
        self.assertAlmostEqual(moving_average([1, 2, 3, 4, 5], 3), 4.0)

    def test_insufficient_data(self):
        self.assertIsNone(moving_average([1, 2], 5))


class TestPercentChange(unittest.TestCase):
    def test_basic(self):
        self.assertAlmostEqual(percent_change(100, 110), 0.1)

    def test_none_input(self):
        self.assertIsNone(percent_change(None, 110))


class TestNormalCdf(unittest.TestCase):
    def test_zero(self):
        self.assertAlmostEqual(normal_cdf(0.0), 0.5, places=2)


class TestClamp(unittest.TestCase):
    def test_clamp(self):
        self.assertEqual(clamp(5, 0, 10), 5)
        self.assertEqual(clamp(-1, 0, 10), 0)
        self.assertEqual(clamp(15, 0, 10), 10)


if __name__ == "__main__":
    unittest.main()
