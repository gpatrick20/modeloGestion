import unittest
import numpy as np
import pandas as pd
from src.risk_math_common import label, validate, LABELS
from src.finalize_risk_math import evaluate


class Riesgo70Tests(unittest.TestCase):
    def test_boundary(self):
        self.assertEqual(label(np.array([0, 60, 69.999, 70, 100])).tolist(), ['en_riesgo', 'en_riesgo', 'en_riesgo', 'sin_riesgo', 'sin_riesgo'])
        self.assertEqual(LABELS, ['en_riesgo', 'sin_riesgo'])

    def test_invalid_data(self):
        for value in [np.nan, np.inf, 101, -1]:
            with self.assertRaises(ValueError):
                validate(pd.DataFrame({'AttendanceRate': [.8], 'StudyHours': [1], 'TestScore_Math': [value]}), True)

    def test_binary_metrics(self):
        table, summary = evaluate(np.array([[8, 2], [3, 7]]))
        self.assertEqual(len(table), 2)
        self.assertAlmostEqual(summary['accuracy'], .75)
        self.assertAlmostEqual(summary['recall_en_riesgo'], .8)
        self.assertAlmostEqual(summary['precision_en_riesgo'], 8 / 11)
        self.assertEqual(summary['en_riesgo_a_verde'], 2)


if __name__ == '__main__':
    unittest.main()
