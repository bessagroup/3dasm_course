"""
Created on October 20, 2023
Helper functions for the L2O project
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from abc import abstractmethod
from typing import Iterable, List, Optional, Tuple, Type

# Third-party
import matplotlib.pyplot as plt
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

    @abstractmethod
    def __call__(self, xarr: PerformanceDataset) -> StrategyDataArray:
        ...


class CustomStrategy(Strategy):
    """Create a strategy of the optimizer to use given features
    of the benchmark problem
    """
    name: str = "custom_strategy"

    @abstractmethod
    def predict(self, features: xr.Dataset) -> Iterable[str]:
        """
        Method to predict the optimizer to use given features of the
        benchmark problem. THis method needs to be implemented by the user.
        Parameters
        ----------
        features : xr.Dataset
            Features of the benchmark problem

        Returns
        -------
        Iterable[str]
            Optimizer to be used for each of the test problems
        Raises
        ------
        NotImplementedError
            If the method is not implemented by the user
        """
        ...

    def __call__(self, xarr: PerformanceDataset) -> StrategyDataArray:
        allowed_features = xarr.drop(['perf_profile', 'ranking'])
        predictions = self.predict(allowed_features)
        return xr.DataArray(predictions, dims=['itemID'],
                            coords={'itemID': xarr['itemID']})


#                                                     Standard strategy classes
# =============================================================================

class WorstPerfProfile(Strategy):
    name: str = 'worst_perf_profile'

    def __call__(self, xarr: PerformanceDataset) -> StrategyDataArray:
        return xarr['perf_profile'].mean('realization').idxmax(
            'optimizer', fill_value='RandomSearch').sel(
                output_dim='y').drop('output_dim')


class BestPerfProfile(Strategy):
    name: str = 'best_perf_profile'

    def __call__(self, xarr: PerformanceDataset) -> StrategyDataArray:
        return xarr['perf_profile'].mean('realization').idxmin(
            'optimizer', fill_value='RandomSearch').sel(
                output_dim='y').drop('output_dim')

# =============================================================================


def create_strategy_xarr(combined_data: PerformanceDataset) -> StrategyDataset:
    return xr.Dataset({strategy.name: strategy for strategy in [
        xr.DataArray(opt, dims=['itemID'],
                     coords={'itemID': combined_data['itemID']},
                     name=opt) for opt in combined_data['optimizer'].values]})

# =============================================================================


class StrategyManager:
    """
    Class to manage the strategies of the optimizer to use given features
    """

    def __init__(self, data: ExperimentData | xr.Dataset,
                 strategy_list: Optional[List[CustomStrategy]] = None):
        """
        Parameters
        ----------

        data : ExperimentData | xr.Dataset`
            ExperimentData object or post-processed data xarray
        strategy_list : List[CustomStrategy], optional
            List of strategies to use
        """
        # Configure the strategies
        if isinstance(data, ExperimentData):
            self.combined_data = open_all_datasets_post(data)

        elif isinstance(data, xr.Dataset):
            self.combined_data = data

        self.strategies = create_strategy_xarr(self.combined_data)

        if strategy_list is None:
            strategy_list = []

        strategy_list.extend(
            [BestPerfProfile(), WorstPerfProfile()])
        self._add_strategies(strategy_list)

    def _add_strategies(self, strategy_list: List[Strategy]):
        for strategy in strategy_list:
            self.strategies[strategy.name] = strategy(self.combined_data)

    def plot(self, title: Optional[str] = None) -> \
            Tuple[plt.Figure, plt.Axes]:
        """
        Plot the performance profiles of the strategies

        Parameters
        ----------
        title : Optional[str]
            Title of the plot

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes of the plot
        """
        xr_f = self.compute_performance_profiles()

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 5))
        for strategy in xr_f.strategy:
            f_one = xr_f.sel(f=1.0, strategy=strategy).values[0]
            ax.plot(xr_f.f, xr_f.sel(strategy=strategy),
                    label=f"{strategy.values} = {f_one}", zorder=-1)
            ax.scatter(1.0, xr_f.sel(f=1.0, strategy=strategy),
                       marker="x", zorder=1)

        if title is not None:
            ax.set_title(title)
        ax.set_xlabel("Performance ratio (f)")
        ax.set_ylabel("Fraction of problems solved")
        ax.legend()
        return fig, ax

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

        xr_f = xr.concat([(pp <= f).mean(dim='problem')
                          for f in f_values],
                         dim=xr.DataArray(f_values, dims='f'))

        return xr_f


#                                                                Plotting tools
# =============================================================================


def plot_perf_profile(data: ExperimentData | xr.Dataset,
                      title: Optional[str] = None) -> \
        Tuple[plt.Figure, plt.Axes]:
    """
    Plot the performance profiles of the strategies

    Parameters
    ----------

    data : ExperimentData | xr.Dataset
        ExperimentData object or xr.Dataset containing the performance profiles
        of the strategies
    title : Optional[str]
        Title of the plot

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes of the plots
    """
    if isinstance(data, ExperimentData):
        data = open_all_datasets_post(data)

    pp = data.stack({'problem': ['itemID', 'realization']})['perf_profile']
    f_values = np.linspace(1, 5, 300)

    xr_f = xr.concat([(pp <= f).mean(dim='problem')
                     for f in f_values], dim=xr.DataArray(f_values, dims='f'))

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    for optimizer in xr_f.optimizer:
        f_one = xr_f.sel(f=1.0, optimizer=optimizer).values[0]
        ax.plot(xr_f.f, xr_f.sel(optimizer=optimizer),
                label=f"{optimizer.values} = {f_one}", zorder=-1)
        ax.scatter(1.0, xr_f.sel(f=1.0, optimizer=optimizer),
                   marker="x", zorder=1)

    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("Performance ratio (f)")
    ax.set_ylabel("Fraction of problems solved")
    ax.legend()
    return fig, ax
