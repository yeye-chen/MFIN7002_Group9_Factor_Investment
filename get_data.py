import yfinance as yf
import bs4 as bs
import requests
import pandas as pd


def save_sp500_tickers():
    url = "https://yfiua.github.io/index-constituents/constituents-sp500.csv"
    df = pd.read_csv(url)
    return df


tickers = pd.read_csv('tickers2trade.csv')

# price and value
data_li = []


for i, ticker in enumerate(tickers['Symbol']):

    print(ticker, i)
    if ticker == 'BRK.B':
        ticker = 'BRK-B'
    # data = yf.download(ticker, start="2024-12-23", end="2024-12-30", interval='1m').reset_index()
    data = yf.download(ticker, start="2024-12-20", end="2024-12-30").reset_index()
    data.columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
    data['Ticker'] = ticker
    data.to_pickle(f"min_data/{ticker}_1230.pkl")
    data_li.append(data)

all_data = pd.concat(data_li)
all_data['Date'] = pd.to_datetime(all_data['Date'])
all_data['Adjustment Factor'] = all_data['Adj Close'] / all_data['Close']
all_data['Adj Open'] = all_data['Adjustment Factor'] * all_data['Open']
all_data['Adj High'] = all_data['Adjustment Factor'] * all_data['High']
all_data['Adj Low'] = all_data['Adjustment Factor'] * all_data['Low']


# tt = pd.read_csv('tickers2trade.csv')
# tt.rename({'Symbol': 'Ticker'}, axis=1, inplace=True)
# all_data = pd.merge(all_data, tt, on='Ticker')

# all_data.to_pickle('data/all_data.pkl')
all_data.to_pickle('min_data/all_min_data_1230.pkl')

# # index data
# sp500_data = yf.download('^GSPC', start="2022-01-01", end="2024-12-19").reset_index()
# sp500_data.columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
# sp500_data['return'] = sp500_data['Adj Close'] / sp500_data['Adj Close'].shift() - 1
# sp500_data.dropna(inplace=True)
# sp500_data['Date'] = pd.to_datetime(sp500_data['Date'])
# sp500_data.to_pickle('data/sp_500.pkl')


