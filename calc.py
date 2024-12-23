import pandas as pd
from tqdm import tqdm
import statsmodels.api as sm

def calculate_beta(high_low_minute_data, mode='full'):
  '''
  Calculate beta as the delta of high_t and low_t
  :param high_low_minute_data: DataFrame (stacked), high and low minute data, need to have columns ['Date', 'Ticker', 'High', 'Low']
  :param mode: str, 'full' or 'opening' or 'closing'; 'full' for full day, 'opening' first 50-min, 'closing' for last 50-min, in a day
  :return: dict, beta_dict containing date, ticker, and the result beta
  '''
  all_tickers = high_low_minute_data['Ticker'].unique()

  output_dict = {'Date': [], 'Ticker': [], 'betas': []}

  for ticker in tqdm(all_tickers, total=len(all_tickers), desc='Calculating betas'):
    if mode == 'full':
      beta_dict = calculate_beta_per_ticker(high_low_minute_data, ticker)
    else:
      beta_dict = calculate_beta_opening_closing_hour_per_ticker(high_low_minute_data, ticker, time=mode)

    output_dict['Date'].extend(beta_dict['Date'])
    output_dict['Ticker'].extend(beta_dict['Ticker'])
    output_dict['betas'].extend(beta_dict['betas'])
    print(f'{ticker} done')

  output_df = pd.DataFrame(output_dict)
  output_df['Date'] = pd.to_datetime(output_df['Date'])

  return output_df

def calculate_beta_per_ticker(high_low_minute_data, ticker):
  '''
  Helper function: calculate beta of a target ticker
  :param high_low_minute_data: DataFrame (stacked), high and low minute data
  :param ticker: str, target ticker
  :return: dict, beta_dict containing date, ticker, and the result beta
  '''

  beta_dict = {'Date': [], 'Ticker': [], 'betas': []}
  hl_df_curr_ticker = high_low_minute_data[high_low_minute_data['Ticker'] == ticker]
  hl_df_per_day = hl_df_curr_ticker.groupby(pd.Grouper(key='Date', freq='D'))

  for date, df_day in hl_df_per_day:
    hl_df_per_hour = df_day.groupby(pd.Grouper(key='Date', freq='h'))
    for hour, df_hour in hl_df_per_hour:

      # y = high_t
      y = df_hour['High']

      # x = low_t
      x = df_hour['Low']
      x = sm.add_constant(x)

      try:
        # regression
        result = sm.OLS(y, x).fit()
        beta = result.params['Low']
        beta_dict['Date'].append(date.date())
        beta_dict['Ticker'].append(ticker)
        beta_dict['betas'].append(beta)
      except Exception as e:
        # reason for exception is unkonwn
        # current grouped hour has no data -> all missing value is 14:00
        print(f'Error: {e}')
        print(f'Ticker: {ticker}, Date: {date}, hour: {hour}')
        print('skip this hour')

  return beta_dict

def calculate_beta_opening_closing_hour_per_ticker(high_low_combined, ticker, time='opening'):
  '''
  Calculate beta as the delta of high_t and low_t, this time only consider opening/ending 50 minutes of a day
  (opening 50-min may contain significant information, in our opinion, rest of the day is more likely to be noise)
  
  :param high_low_minute_data: DataFrame (stacked), high and low minute data
  :param ticker: str, target ticker
  :return: DataFrame, ['Date', 'Ticker', 'betas']
  '''
  beta_dict = {'Date': [], 'Ticker': [], 'betas': []}
  hl_df_curr_ticker = high_low_combined[high_low_combined['Ticker'] == ticker]
  
  hl_df_curr_ticker['day'] = hl_df_curr_ticker['Date'].dt.date
  hl_df_per_day = hl_df_curr_ticker.groupby(pd.Grouper(key='day'))

  for date, df_day in hl_df_per_day:
    
    period_df = None
    if time == 'opening':
      period_df = df_day.head(50)
    else:
      period_df = df_day.tail(50)

    try:
      # opening hour beta
      y = period_df['High']
      x = period_df['Low']
      x = sm.add_constant(x)
      result1 = sm.OLS(y, x).fit()
      beta = result1.params['low']

    except Exception as e:
      print(f'Error: {e}')
      print(f'Ticker: {ticker}, Date: {date}')
      print('skip this day')
      continue

    beta_dict['Date'].append(date)
    beta_dict['Ticker'].append(ticker)
    beta_dict['betas'].append(beta)

  beta_df = pd.DataFrame(beta_dict)
  beta_df['Date'] = pd.to_datetime(beta_df['Date'])
  return beta_dict