"""Type stub file for TimeSeriesTools package."""

from typing import Dict, List, Optional, Tuple, Union, Any
from pandas import DataFrame, Series

# Version information
__version__: str
__author__: str
__author_email__: str
__license__: str

# Classes
class Analyze:
    def __init__(
        self,
        verbose: bool = ...,
        stationarity_config: Optional[Dict[str, Any]] = ...,
        causality_config: Optional[Dict[str, Any]] = ...
    ) -> None: ...

    def analyze_correlation(
        self,
        x: DataFrame,
        y: Series,
        decompose: bool = ...
    ) -> Dict[str, Any]: ...

class AnalyzeCorrelation:
    def __init__(
        self,
        x: DataFrame,
        y: Series,
        decompose: bool = ...,
        verbose: bool = ...,
        stationarity_config: Optional[Dict[str, Any]] = ...,
        causality_config: Optional[Dict[str, Any]] = ...
    ) -> None: ...

class CausalityAnalyzer:
    def __init__(
        self,
        causality_config: Optional[Dict[str, Any]] = ...,
        verbose: bool = ...
    ) -> None: ...

class StationaryTests:
    def __init__(
        self,
        test_config: Optional[Dict[str, Any]] = ...,
        verbose: bool = ...
    ) -> None: ...

# Functions
def test_data1(
    path_to_src: Optional[str] = ...,
    return_xy: bool = ...
) -> Union[DataFrame, Tuple[DataFrame, Series]]: ...

def test_data2(
    return_xy: bool = ...
) -> Union[DataFrame, Tuple[DataFrame, Series]]: ...

def test_data3(
    start_date: str = ...,
    return_xy: bool = ...,
    target: str = ...
) -> Union[DataFrame, Tuple[DataFrame, Series]]: ...

def random_test_data(
    n: int = ...,
    start_date: str = ...,
    return_xy: bool = ...
) -> Union[DataFrame, Tuple[DataFrame, Series]]: ...

# Default configurations
default_config: Dict[str, Dict[str, Any]]