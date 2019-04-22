import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import quandl
import scipy.optimize as sco
from random import randint

import config
quandl.ApiConfig.api_key = config.api_key

def get_historical_price(tickers, start_date, end_date):
    data = quandl.get_table('WIKI/PRICES', ticker=tickers,
        qopts={ 'columns': ['date', 'ticker', 'adj_close'] },
        date={ 'gte': start_date, 'lte': end_date }, paginate=True)
    df = data.set_index('date')
    table = df.pivot(columns='ticker')
    table.columns = [col[1] for col in table.columns]
    return table

def portfolio_annualized_performance(weights, mean_returns, cov_matrix):
    port_return = np.sum(mean_returns*weights) * 252
    port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) \
        * np.sqrt(252)
    return port_return, port_std

def random_portfolios(num_portfolios, mean_returns, cov_matrix,
                      risk_free_rate):
    results = np.zeros((3,num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        weights_record.append(weights)
        port_return, port_std = portfolio_annualized_performance(
            weights, mean_returns, cov_matrix)
        results[0][i] = port_return
        results[1][i] = port_std
        results[2][i] = (port_return - risk_free_rate) / port_std
    return results, weights_record

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    port_return, port_std = portfolio_annualized_performance(
        weights, mean_returns, cov_matrix)
    return - (port_return - risk_free_rate) / port_std

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_tickers = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for ticker in range(num_tickers))
    result = sco.minimize(neg_sharpe_ratio,
        num_tickers*[1./num_tickers],
        args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def portfolio_standard_deviation(weights, mean_returns, cov_matrix):
    return portfolio_annualized_performance(
        weights, mean_returns, cov_matrix)[1]

def min_standard_deviation(mean_returns, cov_matrix):
    num_tickers = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for ticker in range(num_tickers))
    result = sco.minimize(portfolio_standard_deviation,
        num_tickers*[1./num_tickers],
        args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def max_sharpe_given_constraints(mean_returns, cov_matrix, bound_constraints):
    num_tickers = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    # constraints = [
    #     {
    #         'type': 'eq',
    #         'fun': lambda weights: np.sum(weights) - 1
    #     },
    #     # {
    #     #     'type': 'ineq',
    #     #     'fun': lambda weights: weights[3] - 0.8
    #     # }
    # ]
    constraints = []
    constraints.append({
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    })
    # for bound_constraint in bound_constraints:
    #     constraints.append(bound_constraint)
    constraints.extend(bound_constraints)
    print(constraints)
    bound = (0.0, 1.0)
    bounds = tuple(bound for ticker in range(num_tickers))
    result = sco.minimize(neg_sharpe_ratio,
        num_tickers*[1./num_tickers],
        args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def get_allocation(columns, weight):
    allocation = pd.DataFrame(
        weight, index=columns, columns=['allocation'])
    allocation.allocation = \
        [round(i*100,2) for i in allocation.allocation]
    allocation = allocation.T
    return allocation

def efficient_return(mean_returns, cov_matrix, target):
    num_tickers = len(mean_returns)
    args = (mean_returns, cov_matrix)

    def portfolio_return(weights):
        return portfolio_annualized_performance(
            weights, mean_returns, cov_matrix)[0]

    constraints = (
        {
            'type': 'eq',
            'fun': lambda weight: portfolio_return(weight) - target
        },
        {
            'type': 'eq',
            'fun': lambda weight: np.sum(weight) - 1
        }
    )
    bounds = tuple((0,1) for ticker in range(num_tickers))
    result = sco.minimize(portfolio_standard_deviation,
        num_tickers*[1./num_tickers],
        args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def efficient_frontier(mean_returns, cov_matrix, returns_range):
    efficients = []
    for ret in returns_range:
        efficients.append(efficient_return(mean_returns, cov_matrix, ret))
    return efficients

def display_calculated_ef_with_random(columns, mean_returns, cov_matrix,
                                      num_portfolios, risk_free_rate,
                                      bound_constraints):
    results, _ = random_portfolios(
        num_portfolios, mean_returns, cov_matrix, risk_free_rate)

    max_sharpe_weight = max_sharpe_ratio(
        mean_returns, cov_matrix, risk_free_rate)['x']
    max_sharpe_port_return, max_sharpe_port_std = \
        portfolio_annualized_performance(
        max_sharpe_weight, mean_returns, cov_matrix)
    max_sharpe_allocation = get_allocation(columns, max_sharpe_weight)

    min_std_weight = min_standard_deviation(
        mean_returns, cov_matrix)['x']
    min_std_port_return, min_std_port_std = \
        portfolio_annualized_performance(
        min_std_weight, mean_returns, cov_matrix)
    min_std_allocation = get_allocation(columns, min_std_weight)

    max_sharpe_given_constraints_weight = max_sharpe_given_constraints(
        mean_returns, cov_matrix, bound_constraints)['x']
    max_sharpe_given_constraints_port_return, \
        max_sharpe_given_constraints_port_std = \
        portfolio_annualized_performance( \
        max_sharpe_given_constraints_weight, mean_returns, cov_matrix)
    max_sharpe_given_constraints_allocation = get_allocation(
        columns, max_sharpe_given_constraints_weight)

    print("-"*80)
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print("Annualized Return:", round(max_sharpe_port_return, 4))
    print("Annualized Standard Deviation:", round(max_sharpe_port_std, 4))
    print("\n")
    print(max_sharpe_allocation)
    print("-"*80)
    print("Maximum Sharpe Ratio Given Constraints Portfolio Allocation\n")
    print("Annualized Return:",
        round(max_sharpe_given_constraints_port_return, 4))
    print("Annualized Standard Deviation:",
        round(max_sharpe_given_constraints_port_std, 4))
    print("\n")
    print(max_sharpe_given_constraints_allocation)
    print("-"*80)
    print("Minimum Standard Deviation Portfolio Allocation\n")
    print("Annualized Return:", round(min_std_port_return, 4))
    print("Annualized Standard Deviation:", round(min_std_port_std, 4))
    print("\n")
    print(min_std_allocation)

    color_map = matplotlib.colors.LinearSegmentedColormap.from_list(
        '', ['#D1D1D1', '#60D68C', '#394EC4'])
    plt.figure(figsize=(10, 7))
    plt.scatter(results[1,:], results[0,:], c=results[2,:],
        cmap=color_map, marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(max_sharpe_port_std, max_sharpe_port_return,
        marker='o', color='#C4396C', edgecolors='black', linewidth=3,
        s=225, zorder=2, label='Maximum Sharpe Ratio')
    plt.scatter(max_sharpe_given_constraints_port_std,
        max_sharpe_given_constraints_port_return,
        marker='o', color='#ECFA52', edgecolors='black', linewidth=3,
        s=225, zorder=2, label='Maximum Sharpe Ratio Given Constraints')
    plt.scatter(min_std_port_std, min_std_port_return,
        marker='o', color='#36A9D6', edgecolors='black', linewidth=3,
        s=225, zorder=2, label='Minimum Standard Deviation')

    target = np.linspace(min_std_port_return, np.max(results[0]), 50)
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
    plt.plot([p['fun'] for p in efficient_portfolios], target,
        linestyle='-', color='black', linewidth=3, zorder=1,
        label='Efficient Frontier')
    plt.title('Calculated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('Annualized Standard Deviation')
    plt.ylabel('Annualized Return')
    plt.legend(labelspacing=1.25, borderpad=1)
    # plt.xlim([0, np.max(results[1])])
    # plt.ylim([0, np.max(results[0])])
    plt.show()

def get_bound_constraints(ticker_position, bound_constraint_dictionary):
    constraints = []
    for index, (ticker, constraint) in enumerate(bound_constraint_dictionary.items()):
    # for ticker, constraint in bound_constraint_dictionary.items():
        if constraint[0] >= 0:
            constraints.append({
                'type': 'ineq',
                # 'fun': lambda weights: weights[index] - bound_constraint_dictionary[ticker][0]
                'fun': lambda weights: weights[ticker_position[ticker]] - bound_constraint_dictionary[ticker][0]
            })
        if constraint[1] >= 0:
            constraints.append({
                'type': 'ineq',
                'fun': lambda weights: - weights[index] + bound_constraint_dictionary[ticker][1]
            })
    return constraints
#
# def get_bound_constraints(tickers, bound_constraint_dictionary):
#     ticker_position = {}
#     constraint_lists = [[-1, -1]] * len(tickers)
#     for index, value in enumerate(tickers):
#         ticker_position[value] = index
#     for ticker, constraint in bound_constraint_dictionary.items():
#         position = ticker_position[ticker]
#         constraint_lists[position] = constraint
#
#     variables = {}
#     constraints = []
#     for index, constraint in enumerate(constraint_lists):
#         variables['index{0}'.format(index)] = index
#         if constraint[0] >= 0:
#             variables['lower_bound_constraint{0}'.format(index)] = \
#                 constraint[0]
#             constraints.append({
#                 'type': 'ineq',
#                 'fun': lambda weights: weights[variables['index{0}'.format(index)]] - variables['lower_bound_constraint{0}'.format(index)]
#             })
#         if constraint[1] >= 0:
#             constraints.append({
#                 'type': 'ineq',
#                 'fun': lambda weights: - weights[c] + d
#             })
#     return constraints

if __name__ == '__main__':
    tickers = ['AAPL', 'AMZN', 'FB', 'GOOGL', 'INTC']
    # tickers = ['AAPL', 'GOOGL', 'FB', 'INTC', 'AMZN']
    tickers.sort() # sort for prevent bound constraint bug
    ticker_position = {}
    for index, value in enumerate(tickers):
        ticker_position[value] = index
    start_date = '2013-1-1'
    end_date = '2017-12-29'
    df = get_historical_price(tickers, start_date, end_date)

    bound_constraint_dictionary = {
        'AAPL': [0.35, -1],
        # 'AMZN': [0.25, -1],.
        # 'GOOGL': [0.3, -1],
        # 'INTC': [0.44, -1]
    }
    bound_constraints = get_bound_constraints(
        ticker_position, bound_constraint_dictionary)

    daily_returns = df.pct_change()
    mean_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()

    annualized_return = pd.DataFrame(
        mean_returns*252, index=df.columns, columns=['annualized return'])
    annualized_return['annualized return'] = \
        [round(i, 4) for i in annualized_return['annualized return']]
    annualized_return = annualized_return.T
    print(annualized_return)

    num_portfolios = 30000
    risk_free_rate = 0.0178

    display_calculated_ef_with_random(df.columns, mean_returns, cov_matrix,
                                      num_portfolios, risk_free_rate,
                                      bound_constraints)
