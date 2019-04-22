import numpy as np

from functions.portfolio import (
    portfolio_return,
    portfolio_variance
)

def mean_returns_function(daily_returns, **kwargs):
    mean_returns = daily_returns.mean()
    return mean_returns

def create_views_and_link_matrix(names, views):
    """
        # input example
        names = ['AAPL', 'AMZN', 'FB', 'GOOGL', 'INTC']
        views = {
            ('AAPL', '>', 'FB', '0.05'),
            ('GOOGL', '=', '0.35')
        }
    """
    # Q = [views[i][3]/252 for i in range(len(views))]  # view matrix
    Q = []
    for view in views:
        isRelativeView = view[1] == '<' or view[1] == '>'
        isAbsoluteView = view[1] == '='
        if (isRelativeView):
            Q.append(view[3] / 252)
        elif (isAbsoluteView):
            Q.append(view[2] / 252)
        else:
            raise ValueError('incorrect view format')

    P = np.zeros([len(views), len(names)])
    nameToIndex = dict()
    for index, name in enumerate(names):
        nameToIndex[name] = index
    for index, _ in enumerate(views):
        isRelativeView = views[index][1] == '<' or views[index][1] == '>'
        isAbsoluteView = views[index][1] == '='
        if (isRelativeView):
            name1, name2 = views[index][0], views[index][2]
            P[index, nameToIndex[name1]] = +1 if views[index][1] == '>' else -1
            P[index, nameToIndex[name2]] = -1 if views[index][1] == '>' else +1
        elif (isAbsoluteView):
            name = views[index][0]
            P[index, nameToIndex[name]] = 1
    return np.array(Q), P

def equilibriumExcessReturns(
                    market_cap, returns, cov_matrix, risk_free_rate):
    weights = market_cap / np.sum(market_cap)
    port_return = portfolio_return(weights, returns, cov_matrix)
    port_var = portfolio_variance(weights, returns, cov_matrix)
    riskAversion = (port_return - risk_free_rate) / port_var
    pi = np.dot(np.dot(riskAversion, cov_matrix), weights)
    return pi

def equilibriumExcessReturnsAdjustedViews(pi, cov_matrix, Q, P):
    tau = .025
    omega = np.dot(np.dot(np.dot(tau, P), cov_matrix), np.transpose(P))

    sub_a = np.linalg.inv(np.dot(tau, cov_matrix))
    sub_b = np.dot(np.dot(np.transpose(P), np.linalg.inv(omega)), P)
    sub_c = np.dot(np.linalg.inv(np.dot(tau, cov_matrix)), pi)
    sub_d = np.dot(np.dot(np.transpose(P), np.linalg.inv(omega)), Q)
    pi_adj = np.dot(np.linalg.inv(sub_a + sub_b), (sub_c + sub_d))
    return pi_adj

def black_litterman_returns_function(daily_returns, **kwargs):
    tickers = kwargs['tickers']
    market_cap = kwargs['market_cap']
    risk_free_rate = kwargs['risk_free_rate']
    views = kwargs['views']

    mean_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()

    # ### shrinkage
    # cov = LedoitWolf().fit(daily_returns.dropna().values)
    # cov_matrix = cov.covariance_

    market_cap_list = []
    for ticker in tickers:
        market_cap_value = market_cap[ticker]
        market_cap_list.append(market_cap_value)

    pi = equilibriumExcessReturns(
        market_cap_list, mean_returns, cov_matrix, risk_free_rate)
    Q, P = create_views_and_link_matrix(tickers, views)
    pi_adj = equilibriumExcessReturnsAdjustedViews(pi, cov_matrix, Q, P)
    return pi_adj
