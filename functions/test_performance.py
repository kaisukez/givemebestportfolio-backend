import numpy as np
import pandas as pd
import math
from datetime import datetime

from .portfolio import (
    portfolio_return,
    portfolio_standard_deviation,
    find_maximum_sharpe_ratio_point
)


def print_testing_result(port_returns, port_std, risk_free_rate):
    print('port return', round(port_returns, 4))
    print('port    std', round(port_std, 4))
    print('port sharpe', round((port_returns - risk_free_rate) / port_std, 4))

def get_allocation(columns, weight, label='allocation'):
    allocation = pd.DataFrame(weight, index=columns, columns=[label])
    allocation[label] = [round(i, 4) for i in allocation[label]]
    allocation = allocation.loc[allocation[label] != 0]
    allocation = allocation.T
    return allocation

def split_dataframe(dataframe, first_half_period=None, second_half_period=None):
    """
    first_half_period = <int> number of days in first half dataframe
    second_half_period = <int> number of days in second half dataframe
    """
    if ((first_half_period != None and second_half_period != None) or \
        (first_half_period == None and second_half_period == None)):
        raise ValueError('Specify only one of [first_half_period, second_half_period].')
    if (first_half_period != None or second_half_period != None):
        if (first_half_period):
            first_half_dataframe = dataframe[:first_half_period]
            second_half_dataframe = dataframe[first_half_period:]
        elif (second_half_period):
            first_half_dataframe = dataframe[:-second_half_period]
            second_half_dataframe = dataframe[-second_half_period:]
    return first_half_dataframe, second_half_dataframe

def get_optimal_weights(dataframe, returns_function, risk_free_rate, **kwargs):
    daily_returns = dataframe.pct_change()
    cov_matrix = daily_returns.cov()

    kwargs['risk_free_rate'] = risk_free_rate
    training_returns = returns_function(daily_returns, **kwargs)

    optimal_weights = find_maximum_sharpe_ratio_point(
        training_returns, cov_matrix, risk_free_rate)['x']
    return optimal_weights

def test_portfolio_performance(dataframe, returns_function, risk_free_rate,
                               training_period=None, testing_period=None,
                               rebalancing_period=None, slicing=False,
                               text='', **kwargs):
    print('-' * 80)
    print(text)
    print('')
    training_df, testing_df = split_dataframe(
        dataframe,
        first_half_period=training_period,
        second_half_period=testing_period
    )

    if (rebalancing_period == None):
        optimal_weights = get_optimal_weights(
            training_df, returns_function, risk_free_rate, **kwargs)
        testing_daily_returns = testing_df.pct_change()
        testing_mean_returns = testing_daily_returns.mean()
        testing_cov_matrix = testing_daily_returns.cov()

        port_returns = portfolio_return(
            optimal_weights, testing_mean_returns, testing_cov_matrix)
        port_std = portfolio_standard_deviation(
            optimal_weights, testing_mean_returns, testing_cov_matrix)

        # buy_date = training_df.index[-1:].strftime('%Y-%m-%d')
        buy_date = training_df.index[-1:].format()[0]
        print(get_allocation(dataframe.columns, optimal_weights, buy_date))
        print('')
        print_testing_result(port_returns, port_std, risk_free_rate)
    else:
        port_returns_result = []
        port_std_result = []
        port_interval_result = [] # how many days when calculating port returns
                                  # or port std
        total_round = math.ceil(len(testing_df) / rebalancing_period)
        for i in range(total_round):
            # split data
            training_df_r, testing_df_r = split_dataframe(
                dataframe,
                first_half_period = len(training_df) + rebalancing_period * i,
            )

            # training / get optimal weights
            if (slicing):
                training_df_r = training_df_r[rebalancing_period * i:]
            testing_interval_df_r = testing_df_r[:rebalancing_period]

            training_df_r = training_df_r.dropna(axis=1)
            testing_interval_df_r = testing_interval_df_r.dropna(axis=1)
            if training_df_r.empty or testing_interval_df_r.empty \
               or len(training_df_r.columns) < 2 \
               or len(testing_interval_df_r.columns) < 2:
                continue
            common_columns = np.intersect1d(training_df_r.columns, testing_interval_df_r.columns)
            # print(common_columns)

            training_df_r = training_df_r[common_columns]
            testing_interval_df_r = testing_interval_df_r[common_columns]
            # print(training_df_r)
            # print(testing_interval_df_r)
            kwargs['tickers'] = common_columns
            optimal_weights_r = get_optimal_weights(
                training_df_r, returns_function, risk_free_rate, **kwargs
            )
            buy_date = training_df_r.index[-1:].format()[0]
            print(get_allocation(common_columns, optimal_weights_r, buy_date))
            print('')


            # testing / get port returns, port std
            port_interval_r = len(testing_interval_df_r)
            testing_daily_returns_r = testing_interval_df_r.pct_change()
            testing_mean_returns_r = testing_daily_returns_r.mean()
            testing_cov_matrix_r = testing_daily_returns_r.cov()

            port_returns_r = portfolio_return(
                optimal_weights_r, testing_mean_returns_r, testing_cov_matrix_r)
            port_std_r = portfolio_standard_deviation(
                optimal_weights_r, testing_mean_returns_r, testing_cov_matrix_r)

            # append result
            port_returns_result.append(port_returns_r)
            port_std_result.append(port_std_r)
            port_interval_result.append(port_interval_r)
        total_port_returns = np.average(
            port_returns_result, weights=port_interval_result)
        total_port_std = np.average(
            port_std_result, weights=port_interval_result
        )
        # total_port_std = combine_std(
        #     port_returns_result, port_std_result, port_interval_result)
        print_testing_result(total_port_returns, total_port_std, risk_free_rate)
