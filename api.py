from flask import Flask, request
import numpy as np
import json
from datetime import datetime, timedelta

from price import (
    get_historical_price_yahoo
)

from functions.efficient_frontier import (
    display_efficient_frontier
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

app = Flask(__name__)

def get_tickers():
    tickers = ['AAL', 'AAPL', 'ADBE', 'ADI', 'ADP', 'ADSK', 'ALGN', 'ALXN', 'AMAT',
        'AMD', 'AMGN', 'AMZN', 'ASML', 'ATVI', 'AVGO', 'BIDU', 'BIIB', 'BKNG',
        'BMRN', 'CDNS', 'CELG', 'CERN', 'CHKP', 'CMCSA', 'COST', 'CSCO', 'CSX',
        'CTAS', 'CTRP', 'CTSH', 'CTXS', 'DLTR', 'EA', 'EBAY', 'EXPE', 'FAST',
        'FISV', 'FOX', 'FOXA', 'GILD', 'GOOG', 'GOOGL', 'HAS', 'HSIC', 'IDXX',
        'ILMN', 'INCY', 'INTC', 'INTU', 'ISRG', 'JBHT', 'KLAC', 'LBTYA',
        'LBTYK', 'LRCX', 'LULU', 'MAR', 'MCHP', 'MDLZ', 'MELI', 'MNST', 'MSFT',
        'MU', 'MXIM', 'MYL', 'NFLX', 'NTAP', 'NTES', 'NVDA', 'ORLY', 'PAYX',
        'PCAR', 'PEP', 'QCOM', 'REGN', 'ROST', 'SBUX', 'SIRI', 'SNPS', 'SWKS',
        'SYMC', 'TMUS', 'TTWO', 'TXN', 'UAL', 'ULTA', 'VRSK', 'VRSN', 'VRTX',
        'WBA', 'WDC', 'WLTW', 'WYNN', 'XEL', 'XLNX', '^NDX']
    return tickers

def get_tickers_without_ndx():
    tickers = get_tickers()
    tickers = tickers[:-1]
    return tickers

def get_dataframe():
    tickers = get_tickers()
    start_date = '2010-1-1'

    df = get_historical_price_yahoo(tickers, start_date, None)
    df = df['Close']

    nasdaq100_df = df[['^NDX']]
    nasdaq100_df = nasdaq100_df.dropna(axis=0)
    # print(nasdaq100_df)

    df = df.dropna(axis=1)
    df = df.drop('^NDX', axis=1)
    # print(df)

    return df, nasdaq100_df

def print_result(weights, daily_returns, cov_matrix, risk_free_rate):
    port_return = portfolio_return(weights, daily_returns.mean(), cov_matrix)
    port_std = portfolio_standard_deviation(weights, daily_returns.mean(), cov_matrix)
    port_sharpe = (port_return - risk_free_rate) / port_std
    # print('return', port_return)
    # print('std   ', port_std)
    # print('sharpe', port_sharpe)
    # print()

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

def what_to_buy(tickers, weights):
    what_to_buy_list = []
    for index, ticker in enumerate(tickers):
        weight = round(weights[index], 4)
        if (weight != 0):
            # print(ticker, weight)
            what_to_buy_list.append([ticker, weight])
    # print()
    return what_to_buy_list

# tickers must be in preset
# risk_free_rate = 0.0 - 1.0
# year = 1, 2, 3, ..., 9
# risk_factor = 0.0 - 1.0
@app.route('/get_optimal_portfolio', methods=['POST'])
def get_optimal_port():
    print('request', request.get_json())
    requested_data = request.get_json()

    tickers = requested_data.get('tickers', get_tickers_without_ndx())
    risk_factor = float(requested_data.get('risk_factor', 0.5))
    years = int(requested_data.get('years', 5))
    risk_free_rate = float(requested_data.get('risk_free_rate', 0.025))

    df, nasdaq100_df = get_dataframe()

    # filter tickers
    df = df[ map(lambda x: x.upper(), tickers) ]

    # filter date
    start_date = str(datetime.now() - timedelta(days=365*years)).split(' ')[0]
    df = df.loc[df.index > start_date]
    nasdaq100_df = nasdaq100_df.loc[df.index > start_date]



    daily_returns_ndx = nasdaq100_df.pct_change()
    cov_matrix_ndx = daily_returns_ndx.cov()

    daily_returns = df.pct_change()
    cov_matrix = daily_returns.cov()

    min_std = find_minimum_standard_deviation_point(daily_returns.mean(), cov_matrix)['x']
    max_sharpe = find_maximum_sharpe_ratio_point(daily_returns.mean(), cov_matrix, risk_free_rate)['x']
    max_return = find_maximum_return_point(daily_returns.mean(), cov_matrix)['x']
    
    # print('nasdaq')
    nasdaq_return, nasdaq_std, nasdaq_sharpe = print_result(np.array([1]), daily_returns_ndx, cov_matrix_ndx, risk_free_rate)

    # print('min std')
    bottom_return, _, _ = print_result(min_std, daily_returns, cov_matrix, risk_free_rate)

    # print('max_sharpe')
    middle_return, _, _ = print_result(max_sharpe, daily_returns, cov_matrix, risk_free_rate)

    # print('max return')
    top_return, _, _ = print_result(max_return, daily_returns, cov_matrix, risk_free_rate)

    target_return = calulate_return_from_risk_factor(risk_factor, bottom_return, middle_return, top_return)
    choosen = find_efficient_return(daily_returns.mean(), cov_matrix, target_return)['x']
    # print('risk_factor', risk_factor)
    choosen_return, choosen_std, choosen_sharpe = print_result(choosen, daily_returns, cov_matrix, risk_free_rate)
    
    what_to_buy_list = what_to_buy(tickers, choosen)
    what_to_buy_list.sort(key = lambda item: item[1])
    what_to_buy_list.reverse()

    response = {
        'nasdaq_index': { 
            'return': nasdaq_return,
            'std': nasdaq_std,
            'sharpe': nasdaq_sharpe
        },
        'portfolio': {
            'return': choosen_return,
            'std': choosen_std,
            'sharpe': choosen_sharpe,
            'what_to_buy': what_to_buy_list
        }
    }

    print('response', json.dumps(response, indent=4))

    return json.dumps(response)