import unittest
import pandas as pd
import numpy as np
from correlation import analyze_correlation  # Replace 'your_module' with the actual module name

class TestAnalyzeCorrelation(unittest.TestCase):

    def setUp(self):
        # Setup method that runs before each test
        self.sample_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randn(100)
        })
        self.x = self.sample_data[['feature1', 'feature2']]
        self.y = self.sample_data['target']

    def test_init(self):
        # Test initialization
        ac = analyze_correlation(self.x, self.y, decompose=False, verbose=False)
        self.assertEqual(ac.cause, 'target')
        self.assertFalse(ac.decmpose)  
        self.assertFalse(ac.verbose)

    def test_setup_period_index(self):
        ac = analyze_correlation(self.x, self.y, verbose=False)
        df_with_index = ac.setup_period_index(self.sample_data)
        self.assertIsInstance(df_with_index.index, pd.PeriodIndex)

    def test_pca_decomposition(self):
        ac = analyze_correlation(self.x, self.y, decompose=True, verbose=False)
        pca_result = ac.pca_decomposition(ac.df_scaled)
        self.assertEqual(pca_result.columns[0], 'PC1')
        self.assertEqual(len(pca_result.columns), 3)  # Assuming 3 components by default

    def test_set_xy(self):
        ac = analyze_correlation(self.x, self.y, decompose=False, verbose=False)
        self.assertEqual(ac.features, ['feature1', 'feature2'])
        self.assertEqual(ac.target, 'target')
        self.assertTrue(ac.df_scaled.equals(ac.scaler.transform(self.x)))

    def test_adf_test(self):
        # Create a known stationary series for testing
        stationary_series = pd.Series(np.random.randn(100)).cumsum()
        ac = analyze_correlation(self.x, self.y, verbose=False)
        p_value = ac.adf_test(stationary_series)
        self.assertLess(p_value, 0.05)  # Assuming we expect the series to be stationary

    def test_check_stationarity(self):
        ac = analyze_correlation(self.x, self.y, decompose=False, verbose=False)
        # Here, we'll check if the method runs without errors as stationarity is harder to test without real data
        try:
            ac.check_stationarity()
        except ValueError as e:
            self.fail(f"check_stationarity raised {e}")

    def test_fit_var_model(self):
        ac = analyze_correlation(self.x, self.y, decompose=False, verbose=False)
        try:
            model = ac._fit_var_model()
            self.assertIsNotNone(model)
        except SystemExit as e:
            self.fail(f"_fit_var_model raised SystemExit: {e}")

    def test_granger_causality(self):
        ac = analyze_correlation(self.x, self.y, decompose=False, verbose=False)
        ac.model = ac._fit_var_model()  # Fit the model first
        results = ac._granger_causality()
        self.assertIsInstance(results, list)

    # Add similar tests for _instantaneous_causality and _contemporaneous_causality

    def test_analyze(self):
        ac = analyze_correlation(self.x, self.y, decompose=False, verbose=False)
        results = ac.analyze()
        self.assertIn('stationarity_tests', results)
        self.assertIn('var_model', results)
        self.assertIn('granger_causality', results)
        self.assertIn('instantaneous_causality', results)
        self.assertIn('contemporaneous_causality', results)

if __name__ == '__main__':
    unittest.main()