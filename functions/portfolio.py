import numpy as np
import scipy.optimize as sco

def portfolio_return(weights, returns, cov_matrix):
    port_return = np.sum(returns * weights) * 252
    return port_return

def negative_portfolio_return(weights, returns, cov_matrix):
    port_return = portfolio_return(weights, returns, cov_matrix)
    return - port_return

def portfolio_standard_deviation(weights, returns, cov_matrix):
    port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) \
        * np.sqrt(252)
    return port_std

def portfolio_variance(weights, returns, cov_matrix):
    port_var = np.dot(weights.T, np.dot(cov_matrix, weights)) * 252
    return port_var

def portfolio_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate):
    port_return = portfolio_return(weights, returns, cov_matrix)
    port_std = portfolio_standard_deviation(weights, returns, cov_matrix)
    port_sharpe = (port_return - risk_free_rate) / port_std
    return port_sharpe

def negative_portfolio_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate):
    port_sharpe = portfolio_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate)
    return - port_sharpe

def portfolio_with_risk_aversion(weights, returns, cov_matrix, risk_aversion):
    port_return = portfolio_return(weights, returns, cov_matrix)
    port_std = portfolio_standard_deviation(weights, returns, cov_matrix)
    port_sharpe = port_return - (risk_aversion / 2) * port_std
    return port_sharpe

def negative_portfolio_with_risk_aversion(weights, returns, cov_matrix, risk_aversion):
    port_sharpe = portfolio_with_risk_aversion(weights, returns, cov_matrix, risk_aversion)
    return - port_sharpe

# def portfolio_annualized_performance(weights, returns, cov_matrix):
#     port_return = portfolio_return(weights, returns, cov_matrix)
#     port_std = portfolio_standard_deviation(weights, returns, cov_matrix)
#     port_var = portfolio_variance(weights, returns, cov_matrix)
#     return port_return, port_std, port_var

def find_maximum_sharpe_ratio_point(returns, cov_matrix, risk_free_rate):
    num_tickers = len(returns)
    args = (returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for ticker in range(num_tickers))
    result = sco.minimize(negative_portfolio_sharpe_ratio,
        num_tickers*[1./num_tickers],
        args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def find_portfolio_with_risk_aversion(returns, cov_matrix, risk_aversion):
    num_tickers = len(returns)
    args = (returns, cov_matrix, risk_aversion)
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for ticker in range(num_tickers))
    result = sco.minimize(negative_portfolio_with_risk_aversion,
        num_tickers*[1./num_tickers],
        args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def find_minimum_standard_deviation_point(returns, cov_matrix):
    num_tickers = len(returns)
    args = (returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for ticker in range(num_tickers))
    result = sco.minimize(portfolio_standard_deviation,
        num_tickers*[1./num_tickers],
        args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def find_maximum_return_point(returns, cov_matrix):
    num_tickers = len(returns)
    args = (returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for ticker in range(num_tickers))
    result = sco.minimize(negative_portfolio_return,
        num_tickers*[1./num_tickers],
        args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def find_efficient_return(returns, cov_matrix, target):
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