import calc
import pandas as pd
import os
import factor_experiments.util as util

data_path = 'data'
factor_path = 'factor_outputs'

if __name__ == '__main__':
  # NOTE: current best factor combo: ['low_vol_ratio_daily', 'beta_daily', 'high_vol_ratio_daily'] using equal weight
  # then add fundamental factor: RETURN_COM_EQY combo 2/3 + ROE * 1/3

  # 1. Calculating beta_daily factor:

  # load data

  high_data_min = pd.read_pickle(os.path.join(data_path, 'high_data_minute.pkl'))
  low_data_min = pd.read_pickle(os.path.join(data_path, 'low_data_minute.pkl'))
  all_data_new_min = pd.read_pickle(os.path.join(data_path, 'all_min_data_1222.pkl'))

  # preprocess to avoid weird timezon issue
  all_data_new_min['Date'] = all_data_new_min['Date'].dt.tz_convert(None)

  # preprocess
  high_data_min = high_data_min.stack().reset_index()
  high_data_min.columns = ['Date', 'Ticker', 'High']
  low_data_min = low_data_min.stack().reset_index()
  low_data_min.columns = ['Date', 'Ticker', 'Low']

  # only need recent two-month data
  high_data_min = high_data_min[(high_data_min['Date'] >= pd.to_datetime('2024-11-01')) & (high_data_min['Date'] <= pd.to_datetime('2024-12-31'))]
  low_data_min = low_data_min[(low_data_min['Date'] >= pd.to_datetime('2024-11-01')) & (low_data_min['Date'] <= pd.to_datetime('2024-12-31'))]
  high_low_combined = pd.merge(high_data_min, low_data_min, on=['Date', 'Ticker'])

  # concat lasest data to old data
  high_low_combined = pd.concat([high_low_combined, all_data_new_min[['Date', 'Ticker', 'High', 'Low']]], ignore_index=True)

  # output result beta_df, which is the beta_daily factor, columns: ['Date', 'Ticker', 'betas']
  beta_df = calc.calculate_beta(high_low_combined, mode='full')

  # TODO 2. calculate low_vol_ratio_daily factor:

  # TODO 3. calculate high_vol_ratio_daily factor:

  # TODO 4. equal weight above factors, using util.factor_weighted():

  # TODO 5. get RETURN_ON_EQUITY factor:
  fund_factor = 'RETURN_COM_EQY'
  fundamental_factors = pd.read_pickle(os.path.join(factor_path, 'fundamental_factors.pkl'))
  fundamental_factors = fundamental_factors['Date', 'Ticker', fund_factor]
  # standardisation
  fundamental_factors[fund_factor] = (fundamental_factors[fund_factor] - fundamental_factors[fund_factor].mean()) / fundamental_factors[fund_factor].std()
  
  # TODO 6. combine using combo 2/3 + RETURN_ON_EQUITY * 1/3 (or maybe 1/4?)

