"""Helper functions for the L2O project"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from abc import abstractmethod
from typing import Any, Dict, List, Type

# Third-party
import numpy as np
import xarray as xr
from f3dasm import ExperimentData

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (martin_van_der_schelling@brown.edu)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================

#                                                             Types & constants
# =============================================================================

# Labels of the output data
RAW_DATASET_LABEL = 'path_raw'
PERFORMANCE_DATASET_LABEL = 'path_post'

# Type aliases
PerformanceDataset = Type[xr.Dataset]
RawDataset = Type[xr.Dataset]
StrategyDataArray = Type[xr.DataArray]
StrategyDataset = Type[xr.Dataset]

#                                                                  Loading data
# =============================================================================


def open_one_dataset_post(experimentdata: ExperimentData,
                          index: int) -> PerformanceDataset:
    """Open the post-processed data of one benchmark problem

    Parameters
    ----------
    experimentdata
        ExperimentData object
    index
        Index of the job to open

    Returns
    -------
        Post-processed data of one benchmark problem in xarray format
    """
    return xr.open_dataset(experimentdata.path /
                           experimentdata.output_data.to_dataframe()
                           [PERFORMANCE_DATASET_LABEL].loc[index])


def open_one_dataset_raw(experimentdata: ExperimentData,
                         index: int) -> RawDataset:
    """Open the raw data of one benchmark problem

    Parameters
    ----------
    experimentdata
        ExperimentData object
    index
        Index of the job to open

    Returns
    -------
        Raw data of one benchmark problem in xarray format
    """
    return xr.open_dataset(experimentdata.path /
                           experimentdata.output_data.to_dataframe()
                           [RAW_DATASET_LABEL].loc[index])


def open_all_datasets_post(experimentdata:
                           ExperimentData) -> PerformanceDataset:
    """Open the post-processed data of all benchmark problems

    Parameters
    ----------
    experimentdata
        ExperimentData object

    Returns
    -------
        Post-processed data of all benchmark problems in xarray format

    Note
    ----
    You need to have the package dask installed in order to load all the
    datasets at once.
    """
    return xr.open_mfdataset([experimentdata.path / path for path in
                              experimentdata.output_data.to_dataframe()
                              [PERFORMANCE_DATASET_LABEL]])

#                                                                Protocol class
# =============================================================================


class Strategy:
    name: str = "strategy"

    def __call__(self, xarr: PerformanceDataset) -> StrategyDataArray:
        ...


class CustomStrategy(Strategy):
    """Create a strategy of the optimizer to use given features
    of the benchmark problem
    """
    name: str = "custom_strategy"

    @abstractmethod
    def predict(self, dim: int, budget: int, noise: int, convex: int,
                separable: int, multimodal: int,
                samples_output: np.ndarray) -> str:
        """
        Method to predict the optimizer to use given features of the
        benchmark problem. THis method needs to be implemented by the user.

        Parameters
        ----------
        dim : int
            Dimension of the benchmark problem
            budget
            Budget of the benchmark problem
        noise : int
            Noise of the benchmark problem
        convex : int
            Convexity of the benchmark problem
        separable : int
            Separability of the benchmark problem
        multimodal : int
            Multimodality of the benchmark problem
        samples_output : np.ndarray
            Output samples of the benchmark problem

        Returns
        -------
        str
            Optimizer to use

        Raises
        ------
        NotImplementedError
            If the method is not implemented by the user
        """
        ...

    def __call__(self, xarr: PerformanceDataset) -> StrategyDataArray:
        predictions = [self.predict(
            **create_features_dict(xarr, id)) for id in xarr.itemID]
        return xr.DataArray(predictions, dims=['itemID'],
                            coords={'itemID': xarr['itemID']})


#                                                     Standard strategy classes
# =============================================================================

class WorstPerfProfile(Strategy):
    name: str = 'worst_perf_profile'

    def __call__(self, xarr: PerformanceDataset) -> StrategyDataArray:
        return xarr['perf_profile'].mean('realization').idxmax(
            'optimizer').sel(output_dim='y').drop('output_dim')


class BestPerfProfile(Strategy):
    name: str = 'best_perf_profile'

    def __call__(self, xarr: PerformanceDataset) -> StrategyDataArray:
        return xarr['perf_profile'].mean('realization').idxmin(
            'optimizer').sel(output_dim='y').drop('output_dim')

# =============================================================================


def create_strategy_xarr(combined_data: PerformanceDataset) -> StrategyDataset:
    return xr.Dataset({strategy.name: strategy for strategy in [
        xr.DataArray(opt, dims=['itemID'],
                     coords={'itemID': combined_data['itemID']},
                     name=opt) for opt in combined_data['optimizer'].values]})


def create_features_dict(combined_data: PerformanceDataset,
                         itemID: int) -> Dict[str, Any]:
    ds = combined_data.drop_vars(['perf_profile', 'ranking'])

    features = {var: ds.sel(itemID=itemID)[var].values for var in ds.data_vars}

    for feature, value in features.items():
        if isinstance(value, np.ndarray):
            if value.shape == ():
                features[feature] = value.item()

        if feature == 'samples_output':
            features[feature] = value.reshape(
                len(combined_data['realization']), -1)

    return features

# =============================================================================


class StrategyManager:
    """
    Class to manage the strategies of the optimizer to use given features
    """

    def __init__(self, experimentdata: ExperimentData,
                 strategy_list: List[CustomStrategy] = None):
        """
        Parameters
        ----------

        experimentdata : ExperimentData
            ExperimentData object
        strategy_list : List[CustomStrategy]
            List of strategies to use
        """
        # Configure the strategies
        self.combined_data = open_all_datasets_post(experimentdata)
        self.strategies = create_strategy_xarr(self.combined_data)

        if strategy_list is None:
            strategy_list = []

        strategy_list.extend(
            [BestPerfProfile(), WorstPerfProfile()])
        self._add_strategies(strategy_list)

    def _add_strategies(self, strategy_list: List[Strategy]):
        for strategy in strategy_list:
            self.strategies[strategy.name] = strategy(self.combined_data)

    def plot(self):
        """
        Plot the performance profiles of the strategies
        """
        self.compute_performance_profiles().plot(hue='strategy')

    def compute_performance_profiles(self) -> xr.DataArray:
        """
        Compute the performance profiles of the strategies

        Returns
        -------
        xr.DataArray
            Performance profiles of the strategies
        """
        perf_profile = xr.concat([self.combined_data.sel(
            itemID=self.strategies['itemID'],
            optimizer=self.strategies[s])
            for s in self.strategies.data_vars],
            dim=xr.DataArray(self.strategies.data_vars, dims='strategy'))

        pp = perf_profile.stack({'problem': ['itemID', 'realization']})[
            'perf_profile']

        # fraction of runs that are within a factor f of the best run
        f_values = np.linspace(1, 5, 300)

        fractions = []
        for f in f_values:
            fractions.append((pp <= f).mean(dim='problem'))

        fractions_xr = xr.DataArray(fractions, dims=('f', 'strategy',
                                                     'output_dim'), coords={
                                    'f': f_values, 'strategy': pp.strategy,
                                    'output_dim': pp.output_dim})

        return fractions_xr


#                                                                Plotting tools
# =============================================================================


def plot_perf_profile(data: ExperimentData | xr.Dataset):
    """
    Plot the performance profiles of the strategies

    Parameters
    ----------

    data : ExperimentData | xr.Dataset
        ExperimentData object or xr.Dataset containing the performance profiles
        of the strategies
    """
    if isinstance(data, ExperimentData):
        data = open_all_datasets_post(data)

    pp = data.stack({'problem': ['itemID', 'realization']})['perf_profile']
    f_values = np.linspace(1, 5, 300)

    fractions = []
    for f in f_values:
        fractions.append((pp <= f).mean(dim='problem'))

    fractions_xr = xr.DataArray(fractions, dims=('f', 'optimizer',
                                                 'output_dim'), coords={
                                'f': f_values, 'optimizer': pp.optimizer,
                                'output_dim': pp.output_dim})
    fractions_xr.plot(hue='optimizer')
