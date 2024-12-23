import pandas as pd
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor

# for ticker in all_tickers:
def calculate_beta(high_low_combined, ticker):
  beta_dict = {'Date': [], 'Ticker': [], 'betas': []}
  hl_df_curr_ticker = high_low_combined[high_low_combined['Ticker'] == ticker]
  hl_df_per_day = hl_df_curr_ticker.groupby(pd.Grouper(key='Date', freq='D'))

  for date, df_day in hl_df_per_day:
    hl_df_per_hour = df_day.groupby(pd.Grouper(key='Date', freq='h'))
    for hour, df_hour in hl_df_per_hour:

      # y = high_t
      y = df_hour['high']

      # x = low_t
      x = df_hour['low']
      x = sm.add_constant(x)

      try:
        result = sm.OLS(y, x).fit()
        beta = result.params['low']
        beta_dict['Date'].append(date.date())
        beta_dict['Ticker'].append(ticker)
        beta_dict['betas'].append(beta)
      except Exception as e:
        # reason for exception is unkonwn, current grouped hour has no data
        # then why group this hour?
        print(f'Error: {e}')
        print(f'Ticker: {ticker}, Date: {date}, hour: {hour}')
        print('skip this hour')

  return beta_dict

def calculate_beta_modified(high_low_combined, ticker):
  beta_dict = {'Date': [], 'Ticker': [], 'betas': []}
  hl_df_curr_ticker = high_low_combined[high_low_combined['Ticker'] == ticker]

  # only take first 50-min and last 50-min to calculate beta
  hl_df_per_day = hl_df_curr_ticker.groupby(pd.Grouper(key='Date', freq='D'))

  for date, df_day in hl_df_per_day:
    opening_hour_df = df_day.head(50)
    closing_hour_df = df_day.tail(50)

    # 1. opening hour beta
    y1 = opening_hour_df['high']
    x1 = opening_hour_df['low']
    x1 = sm.add_constant(x1)
    result1 = sm.OLS(y1, x1).fit()
    beta1 = result1.params['low']

    # 2. closing hour beta
    y2 = closing_hour_df['high']
    x2 = closing_hour_df['low']
    x2 = sm.add_constant(x2)
    result2 = sm.OLS(y2, x2).fit()
    beta2 = result2.params['low']

    # 3. average beta
    beta = (beta1 + beta2) / 2

    beta_dict['Date'].append(date.date())
    beta_dict['Ticker'].append(ticker)
    beta_dict['betas'].append(beta)

  return beta_dict

def loading_data():
  high_data = pd.read_csv('high_data.csv', index_col=0, parse_dates=True)
  low_data = pd.read_csv('low_data.csv', index_col=0, parse_dates=True)

  high_data.ffill()
  low_data.ffill()

  high_data_stack = high_data.stack().reset_index()
  high_data_stack.columns = ['Date', 'Ticker', 'high']

  low_data_stack = low_data.stack().reset_index()
  low_data_stack.columns = ['Date', 'Ticker', 'low']

  high_low_combined = pd.merge(high_data_stack, low_data_stack, on=['Date', 'Ticker'])

  return high_low_combined

def process(high_low_combined, beta_dict_all, all_tickers):

  with ProcessPoolExecutor() as executor:
    futures = {executor.submit(calculate_beta, high_low_combined, ticker): ticker for ticker in all_tickers}
    for future in tqdm(as_completed(futures), total=len(all_tickers), desc='Processing'):
      beta_dict = future.result()
      beta_dict_all['Date'].extend(beta_dict['Date'])
      beta_dict_all['Ticker'].extend(beta_dict['Ticker'])
      beta_dict_all['betas'].extend(beta_dict['betas'])
      print(f'Ticker {futures[future]} completed')

if __name__ == '__main__':
  high_low_combined = loading_data()
  beta_dict_all = {'Date': [], 'Ticker': [], 'betas': []}
  all_tickers = high_low_combined['Ticker'].unique()

  done_beta_df = pd.read_pickle('beta_df.pkl')
  remaining_tickers = list(set(all_tickers) - set(done_beta_df['Ticker'].unique()))
  
  # remaining ticker logic
  beta_df = pd.read_pickle('beta_df.pkl')

  print('--- Calculating Betas ---')
  process(high_low_combined, beta_dict_all, remaining_tickers)

  beta_open_close_hours_avg = pd.DataFrame(beta_dict_all)
  beta_open_close_hours_avg.to_pickle('remaining_beta.pkl')