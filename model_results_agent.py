import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

class ModelResultsAgent:
    """Agent for generating formatted Markdown summaries of time series model results."""
    
    def __init__(self):
        self.markdown_content = []
        
    def add_header(self, text: str, level: int = 1):
        """Add a header to the markdown content."""
        self.markdown_content.append(f"{'#' * level} {text}\n")
        
    def add_section(self, text: str):
        """Add a text section to the markdown content."""
        self.markdown_content.append(f"{text}\n")
        
    def add_table(self, df: pd.DataFrame, title: Optional[str] = None):
        """Convert a pandas DataFrame to markdown table format."""
        if title:
            self.markdown_content.append(f"### {title}\n")
        self.markdown_content.append(df.to_markdown() + "\n")

    def format_var_results(self, var_model, target_variable):
        """Format VAR model results for the target endogenous variable."""
        self.add_header("Vector Autoregression (VAR) Analysis", 2)
        
        if var_model is None:
            self.add_section("No VAR model results available.")
            return
            
        try:
            # Get parameters and statistics for target variable equation
            params = pd.DataFrame()
            params['Coefficient'] = var_model.params[target_variable]
            params['Std Error'] = var_model.bse[target_variable]
            params['t-stat'] = var_model.tvalues[target_variable]
            params['p-value'] = var_model.pvalues[target_variable]
            
            self.add_header(f"Equation Results for {target_variable}", 3)
            self.add_table(params, "Coefficient Statistics")
            
            # Extract model fit statistics
            model_stats = pd.DataFrame({
                'Value': {
                    'R-squared': var_model.fittedvalues[target_variable].corr(var_model.model.endog[:, var_model.model.endog_names.index(target_variable)])**2,
                    'Log Likelihood': var_model.llf,
                    'AIC': var_model.aic,
                    'BIC': var_model.bic,
                }
            })
            self.add_table(model_stats, "Model Statistics")
            
            # Add residual correlation matrix
            resid_corr = pd.DataFrame(
                var_model.resid.corr(),
                columns=var_model.names,
                index=var_model.names
            )
            self.add_header("Residual Correlation Matrix", 3)
            self.add_table(resid_corr)
            
        except Exception as e:
            self.add_section(f"Unable to generate VAR model summary - {str(e)}")
            
    def format_causality_results(self, causality_results):
        """Format Granger causality test results."""
        self.add_header("Causality Analysis", 2)
        
        if not causality_results:
            self.add_section("No causality analysis results available.")
            return
            
        if isinstance(causality_results, dict):
            # Handle dictionary format
            for cause, effects in causality_results.items():
                self.add_section(f"**{cause}** Granger-causes:")
                if isinstance(effects, dict):
                    for effect, p_value in effects.items():
                        significance = "✓" if p_value < 0.05 else "✗"
                        self.add_section(f"- {effect}: {significance} (p-value: {p_value:.4f})")
                else:
                    self.add_section(f"- Result: {effects}")
        elif isinstance(causality_results, list):
            # Handle list format
            for result in causality_results:
                if isinstance(result, dict):
                    cause = result.get('cause', 'Unknown')
                    effect = result.get('effect', 'Unknown')
                    p_value = result.get('p_value', 1.0)
                    significance = "✓" if p_value < 0.05 else "✗"
                    self.add_section(f"- {cause} → {effect}: {significance} (p-value: {p_value:.4f})")
                else:
                    self.add_section(f"- {str(result)}")
        else:
            self.add_section(f"Causality results in unexpected format: {type(causality_results)}")

    def generate_summary(self, results: Dict[str, Any], target_variable: str) -> str:
        """Generate a complete markdown summary of model results."""
        self.markdown_content = []
        
        # Main header
        self.add_header("Time Series Analysis Results")
        
        try:
            # Data Overview
            if 'new_data' in results:
                self.add_header("Data Overview", 2)
                data = results['new_data']
                self.add_section(f"- Number of series: {data.shape[1]}")
                self.add_section(f"- Time period: {data.index[0]} to {data.index[-1]}")
                self.add_section(f"- Number of observations: {len(data)}")
            
            # VAR Model Results
            if 'var_model' in results:
                self.format_var_results(results['var_model'], target_variable)
            
            # Granger Causality Results
            if 'granger_results' in results:
                self.format_causality_results(results['granger_results'])
            
            return "\n".join(self.markdown_content)
            
        except Exception as e:
            error_msg = [
                "# Error in Generating Results Summary",
                f"An error occurred while generating the analysis summary: {str(e)}",
                "\nPlease check the following:",
                "- Input data format",
                "- Model execution status",
                "- Result dictionary structure"
            ]
            return "\n".join(error_msg)
