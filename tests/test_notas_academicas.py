import unittest
import numpy as np
import pandas as pd
from src.experimento_notas_academicas import FEATURES, LABELS, CUT, label, validate, metrics


class NotasAcademicasTests(unittest.TestCase):
    def test_features(self):
        self.assertEqual(FEATURES, ['AttendanceRate', 'StudyHours', 'TestScore_Reading', 'TestScore_Science'])
        self.assertEqual(LABELS, ['en_riesgo', 'sin_riesgo'])
        self.assertEqual(CUT, 70)

    def test_boundary(self):
        self.assertEqual(label(np.array([0, 69.999, 70, 100])).tolist(), ['en_riesgo', 'en_riesgo', 'sin_riesgo', 'sin_riesgo'])

    def test_invalid_values(self):
        for value in [np.nan, np.inf, 101, -1]:
            with self.assertRaises(ValueError):
                validate(pd.DataFrame({'AttendanceRate': [.8], 'StudyHours': [1], 'TestScore_Reading': [value], 'TestScore_Science': [80], 'TestScore_Math': [50]}), True)

    def test_metrics(self):
        summary = metrics(np.array([[8, 2], [3, 7]]))
        self.assertAlmostEqual(summary['accuracy'], .75)
        self.assertAlmostEqual(summary['recall_en_riesgo'], .8)
        self.assertAlmostEqual(summary['precision_en_riesgo'], 8 / 11)
        self.assertEqual(summary['en_riesgo_a_verde'], 2)


if __name__ == '__main__':
    unittest.main()
