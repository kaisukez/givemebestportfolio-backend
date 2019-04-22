import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.colors
from datetime import datetime, timedelta

from price import (
    get_historical_price_yahoo
)

from functions.efficient_frontier import (
    display_efficient_frontier,
    display_calculated_ef_with_random
)

from functions.return_function import (
    mean_returns_function,
    black_litterman_returns_function
)

from functions.test_performance import (
    test_portfolio_performance
)

from functions.portfolio import (
    find_maximum_return_point,
    find_minimum_standard_deviation_point,
    find_maximum_sharpe_ratio_point,
    find_efficient_return,
    find_portfolio_with_risk_aversion,
    portfolio_return,
    portfolio_standard_deviation,
    portfolio_sharpe_ratio
)

def test(tickers):
    # market_cap = np.array([970e9, 837e9, 416e9, 744e9, 219e9])

    start_date = '2000-1-1'
    end_date = '2018-12-31'
    # df = get_historical_price_quandl(tickers, start_date, end_date)
    df = get_historical_price_yahoo(tickers, start_date, end_date)
    df = df['Close']
    df = df.dropna()

    daily_returns = df.pct_change()
    mean_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()
    # num_portfolios = 5000000
    num_portfolios = 30000
    risk_free_rate = 0.0178

    # display_efficient_frontier(mean_returns, cov_matrix,
    #     num_portfolios, risk_free_rate, 'blue', 'mean-variance')
    # # plt.show()
    # plt.savefig('efficient_frontier', dpi=450)

    display_calculated_ef_with_random(None, mean_returns, cov_matrix,
                                      num_portfolios, risk_free_rate)
    plt.savefig('efficient_frontier_2', dpi=450)

    # test_portfolio_performance(df, mean_returns_function,
    #     risk_free_rate, training_period=252, rebalancing_period=None,
    #     text='mean-variance optimization [buy once]')
    # test_portfolio_performance(df, mean_returns_function,
    #     risk_free_rate, training_period=252, rebalancing_period=80, slicing=True,
    #     text='mean-variance optimization [optimize every 4 months]')

def get_nasdaq100_tickers():
    tickers = ["AAL", "AAPL", "ADBE", "ADI", "ADP", "ADSK", "ALGN", "ALXN",
               "AMAT", "AMD", "AMGN", "AMZN", "ASML", "ATVI", "AVGO", "BIDU",
               "BIIB", "BKNG", "BMRN", "CDNS", "CELG", "CERN", "CHKP", "CHTR",
               "CMCSA", "COST", "CSCO", "CSX", "CTAS", "CTRP", "CTSH", "CTXS",
               "DLTR", "EA", "EBAY", "EXPE", "FAST", "FB", "FISV", "FOX",
               "FOXA", "GILD", "GOOG", "GOOGL", "HAS", "HSIC", "IDXX", "ILMN",
               "INCY", "INTC", "INTU", "ISRG", "JBHT", "JD", "KHC", "KLAC",
               "LBTYA", "LBTYK", "LRCX", "LULU", "MAR", "MCHP", "MDLZ", "MELI",
               "MNST", "MSFT", "MU", "MXIM", "MYL", "NFLX", "NTAP", "NTES",
               "NVDA", "NXPI", "ORLY", "PAYX", "PCAR", "PEP", "PYPL", "QCOM",
               "REGN", "ROST", "SBUX", "SIRI", "SNPS", "SWKS", "SYMC", "TMUS",
               "TSLA", "TTWO", "TXN", "UAL", "ULTA", "VRSK", "VRSN", "VRTX",
               "WBA", "WDAY", "WDC", "WLTW", "WYNN", "XEL", "XLNX"]
    return tickers

# if __name__ == '__main__':
#     risk_free_rate = 0.0178
#     df = get_historical_price_set100()
#     print(df)

#     rics = list(df.columns)
#     market_cap_dict = get_mkt_cap_dict(rics)

#     test_portfolio_performance(df, black_litterman_returns_function,
#         risk_free_rate, training_period=252, rebalancing_period=80, slicing=True,
#         text='black-litterman optimization [optimize every 4 months]',
#         tickers=rics, market_cap=market_cap_dict, views=[])

def what_to_buy(tickers, weights):
    for index, ticker in enumerate(tickers):
        weight = round(weights[index], 4)
        if (weight != 0):
            print(ticker, weight)
    print()

def get_start_and_end_date():
    """
    - get start_date and end_date string for requesting data from yahoo
    """
    # start_date = '2011-01-01'
    start_date = str(datetime.now() - timedelta(days=366)).split(' ')[0]
    end_date = str(datetime.now() - timedelta(days=1)).split(' ')[0]
    return start_date, end_date

def print_result(weights, daily_returns, cov_matrix, risk_free_rate):
    port_return = portfolio_return(weights, daily_returns.mean(), cov_matrix)
    port_std = portfolio_standard_deviation(weights, daily_returns.mean(), cov_matrix)
    port_sharpe = (port_return - risk_free_rate) / port_std
    print('return', port_return)
    print('std   ', port_std)
    print('sharpe', port_sharpe)
    print()

    return port_return, port_std, port_sharpe

def calulate_return_from_risk_factor(risk_factor, bottom, middle, top):
    if (risk_factor == 0):
        return bottom
    elif (risk_factor == 0.5):
        return middle
    elif (risk_factor == 1):
        return top
    elif (risk_factor < 0 or risk_factor > 1):
        raise 'risk_factor must be in between 0 and 1'

    if (risk_factor < 0.5):
        risk_factor = risk_factor * 2
        return bottom + (middle - bottom) * risk_factor
    elif (risk_factor > 0.5):
        risk_factor = (risk_factor - 0.5) * 2
        return middle + (top - middle) * risk_factor

# if __name__ == '__main__':
#     start_date = '2010-1-1'
#     # end_date = '2018-12-31'
#     # df = get_historical_price_quandl(tickers, start_date, end_date)
#     tickers = ['AAL', 'AAPL', 'ADBE', 'ADI', 'ADP', 'ADSK', 'ALGN', 'ALXN', 'AMAT',
#        'AMD', 'AMGN', 'AMZN', 'ASML', 'ATVI', 'AVGO', 'BIDU', 'BIIB', 'BKNG',
#        'BMRN', 'CDNS', 'CELG', 'CERN', 'CHKP', 'CMCSA', 'COST', 'CSCO', 'CSX',
#        'CTAS', 'CTRP', 'CTSH', 'CTXS', 'DLTR', 'EA', 'EBAY', 'EXPE', 'FAST',
#        'FISV', 'FOX', 'FOXA', 'GILD', 'GOOG', 'GOOGL', 'HAS', 'HSIC', 'IDXX',
#        'ILMN', 'INCY', 'INTC', 'INTU', 'ISRG', 'JBHT', 'KLAC', 'LBTYA',
#        'LBTYK', 'LRCX', 'LULU', 'MAR', 'MCHP', 'MDLZ', 'MELI', 'MNST', 'MSFT',
#        'MU', 'MXIM', 'MYL', 'NFLX', 'NTAP', 'NTES', 'NVDA', 'ORLY', 'PAYX',
#        'PCAR', 'PEP', 'QCOM', 'REGN', 'ROST', 'SBUX', 'SIRI', 'SNPS', 'SWKS',
#        'SYMC', 'TMUS', 'TTWO', 'TXN', 'UAL', 'ULTA', 'VRSK', 'VRSN', 'VRTX',
#        'WBA', 'WDC', 'WLTW', 'WYNN', 'XEL', 'XLNX', '^NDX']

#     df = get_historical_price_yahoo(tickers, start_date, None)
#     df = df['Close']

#     nasdaq100_df = df[['^NDX']]
#     print(nasdaq100_df)

#     df = df.dropna(axis=1)
#     df = df.drop('^NDX', axis=1)
#     print(df)

#     print()

#     new_df = df[map(lambda x: x.upper(), ['amd', 'ULTA'])]
#     print(new_df.index)

#     start_date_haha = str(datetime.now() - timedelta(days=365*9)).split(' ')[0]
#     print(new_df.loc[new_df.index > start_date_haha])

#     risk_free_rate = 0.025

#     daily_returns_ndx = nasdaq100_df.pct_change()
#     cov_matrix_ndx = daily_returns_ndx.cov()

#     daily_returns = df.pct_change()
#     cov_matrix = daily_returns.cov()
#     # risk_aversion = 2

    
#     # y = find_portfolio_with_risk_aversion(daily_returns.mean(), cov_matrix, risk_aversion)['x']
#     # print_result(y, daily_returns, cov_matrix, risk_free_rate)

#     min_std = find_minimum_standard_deviation_point(daily_returns.mean(), cov_matrix)['x']
#     max_sharpe = find_maximum_sharpe_ratio_point(daily_returns.mean(), cov_matrix, risk_free_rate)['x']
#     max_return = find_maximum_return_point(daily_returns.mean(), cov_matrix)['x']
    
#     print('nasdaq')
#     nasdaq, _, _ = print_result(np.array([1]), daily_returns_ndx, cov_matrix_ndx, risk_free_rate)

#     print('min std')
#     bottom, _, _ = print_result(min_std, daily_returns, cov_matrix, risk_free_rate)

#     print('max_sharpe')
#     middle, _, _ = print_result(max_sharpe, daily_returns, cov_matrix, risk_free_rate)

#     print('max return')
#     top, _, _ = print_result(max_return, daily_returns, cov_matrix, risk_free_rate)

#     risk_factor = 0.8
#     target_return = calulate_return_from_risk_factor(risk_factor, bottom, middle, top)
#     choosen = find_efficient_return(daily_returns.mean(), cov_matrix, target_return)['x']
#     print('risk_factor', risk_factor)
#     print_result(choosen, daily_returns, cov_matrix, risk_free_rate)



if __name__ == '__main__':
    # tickers = ['AAPL', 'AMZN', 'GOOGL', 'INTC']
    # tickers = get_nasdaq100_tickers()
    tickers = ['AAPL', 'ADBE', 'NFLX', 'TSLA']
    test(tickers)

    # print('-' * 80)
    # print('NASDAQ-100')
    # print('')
    # tickers = ['^NDX']
    # start_date = '2016-07-03'
    # end_date = '2018-12-31'
    # # df = get_historical_price_quandl(tickers, start_date, end_date)
    # df = get_historical_price_yahoo(tickers, start_date, end_date)
    # df = df['Close']
    
    # daily_returns = df.pct_change()
    # mean_returns = daily_returns.mean()
    # std_returns = daily_returns.std()
    # risk_free_rate = 0.0178
    # print('NASDAQ-100 return', round(mean_returns, 4))
    # print('NASDAQ-100 std   ', round(std_returns, 4))
    # print('NASDAQ-100 sharpe', round((mean_returns - risk_free_rate) / std_returns, 4))
