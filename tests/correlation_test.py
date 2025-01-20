import unittest
import pandas as pd
import numpy as np
import statsmodels.api as sm

from correlation import analyze_correlation  

class TestAnalyzeCorrelation(unittest.TestCase):

    def setUp(self):
        # Setup method that runs before each test
        macrodata = sm.datasets.macrodata.load_pandas().data
        macrodata.index = pd.period_range('1959Q1', '2009Q3', freq='Q')
        macrodata.index = macrodata.index.to_timestamp()
        macrodata = macrodata.drop(columns = ['year', 'quarter'])
        macrodata = macrodata.diff().dropna()
        x = macrodata.drop(columns = 'realgdp')
        y = macrodata['realgdp']
        self.sample_data = x.merge(y, left_index=True, right_index=True)    
        self.x = x; self.y = y
        self.features = x.columns.tolist()
        self.target = 'realgdp'
       
    def test_init(self):
        # Test initialization
        ac = analyze_correlation(self.x, self.y, decompose=False, verbose=False)
        self.assertEqual(ac.cause, self.target)
        self.assertFalse(ac.decompose)  
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
        self.assertEqual(ac.target, self.target)
        if ac.feature_change == False:
            self.assertEqual(ac.features, self.features)
            scaled = ac.df_scaled
            sc = pd.DataFrame(ac.scaler.fit_transform(self.x), columns=self.features, index=self.x.index)
            sc[self.target] = self.y
            self.assertTrue(scaled.equals(sc))

    def test_adf_test(self):
        # Create a known stationary series for testing
        np.random.seed(0)
        stationary_series = pd.Series(np.random.randn(100)).cumsum()
        ac = analyze_correlation(self.x, self.y, verbose=False)
        p_value = ac.adf_test(stationary_series)
        self.assertGreater(p_value, 0.05)  

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