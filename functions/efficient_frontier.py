import numpy as np
import scipy.optimize as sco
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.colors

from .portfolio import (
    portfolio_return,
    portfolio_standard_deviation,
    portfolio_sharpe_ratio,
    find_minimum_standard_deviation_point,
    find_maximum_sharpe_ratio_point,
    find_maximum_return_point
)

def random_portfolios(num_portfolios, returns, cov_matrix, risk_free_rate):
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(len(returns))
        weights /= np.sum(weights)
        weights_record.append(weights)

        port_return = portfolio_return(weights, returns, cov_matrix)
        port_std = portfolio_standard_deviation(weights, returns, cov_matrix)
        port_sharpe = portfolio_sharpe_ratio(weights, returns, cov_matrix,
                                             risk_free_rate)
        results[0][i] = port_return
        results[1][i] = port_std
        results[2][i] = port_sharpe
    return results, weights_record

def efficient_return(returns, cov_matrix, target):
    num_tickers = len(returns)
    args = (returns, cov_matrix)

    constraints = (
        {
            'type': 'eq',
            'fun': lambda weight: portfolio_return(weight, returns, cov_matrix) - target
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


def efficient_frontier(returns, cov_matrix, target_returns):
    efficients = []
    for target_return in target_returns:
        efficients.append(efficient_return(returns, cov_matrix, target_return))
    return efficients

def display_efficient_frontier(returns, cov_matrix,
                               num_portfolios, risk_free_rate, color, label):
    results, _ = random_portfolios(
        num_portfolios, returns, cov_matrix, risk_free_rate)

    min_std_weight = find_minimum_standard_deviation_point(
        returns, cov_matrix)['x']
    min_std_port_return = portfolio_return(min_std_weight, returns, cov_matrix)
    min_std_port_std = portfolio_standard_deviation(min_std_weight, returns, cov_matrix)

    target_returns = np.linspace(min_std_port_return, np.max(results[0]), 50)
    print('target returns', target_returns)
    efficient_portfolios = efficient_frontier(
        returns, cov_matrix, target_returns)
    plt.plot([p['fun'] for p in efficient_portfolios], target_returns,
        linestyle='-', color=color, linewidth=3, zorder=1,
        label=label)
    plt.title('Efficient Frontier')
    plt.xlabel('Annualized Standard Deviation')
    plt.ylabel('Annualized Return')
    # plt.legend(labelspacing=1.25, borderpad=1)
    # plt.show()

def display_tangent_portfolio(returns, cov_matrix, risk_free_rate, columns,
                              edgecolors, color):
    max_sharpe_weight = max_sharpe_ratio(
        returns, cov_matrix, risk_free_rate)['x']
    max_sharpe_port_return, max_sharpe_port_std, _ = \
        portfolio_annualized_performance(
        max_sharpe_weight, returns, cov_matrix)
    max_sharpe_allocation = get_allocation(columns, max_sharpe_weight)

    print("-" * 80)
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print("Annualized Return:", round(max_sharpe_port_return, 4))
    print("Annualized Standard Deviation:", round(max_sharpe_port_std, 4))
    print("Annualized Sharpe Ratio:", round((max_sharpe_port_return - risk_free_rate)/max_sharpe_port_std, 4))
    print("\n")
    print(max_sharpe_allocation)

    plt.scatter(max_sharpe_port_std, max_sharpe_port_return,
        marker='o', color=color, edgecolors=edgecolors, linewidth=3,
        s=225, zorder=2, label='tangency portfolio')


def display_calculated_ef_with_random(columns, returns, cov_matrix,
                                      num_portfolios, risk_free_rate):
    results, _ = random_portfolios(
        num_portfolios, returns, cov_matrix, risk_free_rate)

    max_sharpe_weight = find_maximum_sharpe_ratio_point(
        returns, cov_matrix, risk_free_rate)['x']
    max_sharpe_port_return = portfolio_return(max_sharpe_weight, returns, cov_matrix)
    max_sharpe_port_std = portfolio_standard_deviation(max_sharpe_weight, returns, cov_matrix)
    # max_sharpe_allocation = get_allocation(columns, max_sharpe_weight)

    min_std_weight = find_minimum_standard_deviation_point(
        returns, cov_matrix)['x']
    min_std_port_return = portfolio_return(min_std_weight, returns, cov_matrix)
    min_std_port_std = portfolio_standard_deviation(min_std_weight, returns, cov_matrix)
    # min_std_allocation = get_allocation(columns, min_std_weight)

    max_return_weight = find_maximum_return_point(
        returns, cov_matrix)['x']
    max_return_port_return = portfolio_return(max_return_weight, returns, cov_matrix)
    max_return_port_std = portfolio_standard_deviation(max_return_weight, returns, cov_matrix)

    # print("-"*80)
    # print("Maximum Sharpe Ratio Portfolio Allocation\n")
    # print("Annualized Return:", round(max_sharpe_port_return, 4))
    # print("Annualized Standard Deviation:", round(max_sharpe_port_std, 4))
    # print("\n")
    # print(max_sharpe_allocation)
    # print("-"*80)
    # print("Minimum Standard Deviation Portfolio Allocation\n")
    # print("Annualized Return:", round(min_std_port_return, 4))
    # print("Annualized Standard Deviation:", round(min_std_port_std, 4))
    # print("\n")
    # print(min_std_allocation)

    color_map = matplotlib.colors.LinearSegmentedColormap.from_list(
        '', ['#D1D1D1', '#60D68C', '#394EC4'])
    plt.figure(figsize=(10, 7))
    plt.scatter(results[1,:], results[0,:], c=results[2,:],
        cmap=color_map, marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(min_std_port_std, min_std_port_return,
        marker='o', color='#36A9D6', edgecolors='black', linewidth=3,
        s=225, zorder=2, label='Minimum Standard Deviation')
    plt.scatter(max_sharpe_port_std, max_sharpe_port_return,
        marker='o', color='#C4396C', edgecolors='black', linewidth=3,
        s=225, zorder=2, label='Maximum Sharpe Ratio')
    plt.scatter(max_return_port_std, max_return_port_return,
        marker='o', color='#E3F539', edgecolors='black', linewidth=3,
        s=225, zorder=2, label='Maximum Return')

    target_returns = np.linspace(min_std_port_return, np.max(results[0]), 50)
    efficient_portfolios = efficient_frontier(returns, cov_matrix, target_returns)
    plt.plot([p['fun'] for p in efficient_portfolios], target_returns,
        linestyle='-', color='black', linewidth=3, zorder=1,
        label='Efficient Frontier')
    plt.title('Simulate Possible Portfolio with Efficient Frontier')
    plt.xlabel('Annualized Standard Deviation')
    plt.ylabel('Annualized Return')
    plt.legend(labelspacing=1.25, borderpad=1)
    # plt.xlim([0, np.max(results[1])])
    # plt.ylim([0, np.max(results[0])])
    # plt.show()