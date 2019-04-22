import pandas as pd
import fix_yahoo_finance as yf
from pathlib import Path
from hashlib import sha256
import base64
import traceback
import shutil

from config import (
    CACHED_DATA_FOLDER
)

yf.pdr_override()

def get_file_name(tickers, start_date, end_date):
    file_name = (str(tickers) + str(start_date) + str(end_date)).encode()
    file_name = sha256(file_name).digest()
    file_name = base64.b64encode(file_name).decode()
    file_name = file_name.replace('/', '')
    file_name = file_name.replace('+', '')
    file_name = file_name.replace('=', '')
    file_name += '.pkl'
    return file_name

def get_historical_price_yahoo(tickers, start_date, end_date):
    # while True:
        # try:
    Path(CACHED_DATA_FOLDER).mkdir(parents=True, exist_ok=True)
    file_name = get_file_name(tickers, start_date, end_date) 
    file_path = (CACHED_DATA_FOLDER / file_name).resolve()
    if Path(file_path).is_file():
        data = pd.read_pickle(file_path)
    else:
        data = yf.download(tickers, start_date, end_date, auto_adjust=True)
        # shutil.rmtree(CACHED_DATA_FOLDER)
        # Path(CACHED_DATA_FOLDER).mkdir(parents=True, exist_ok=True)
        data.to_pickle(file_path)
    return data
        # except ValueError as e:
        #     print(e)
