import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import alphalens as al
import seaborn as sns
import statsmodels.api as sm
import math
import os

def create_factor_graphs(factor_data, price_data, quantiles=20, max_loss=0.4):
  '''
  This function creates factor graphs using Alphalens library.
  factor_data and price_data should be aligned and have the same index/same frequency.
  '''

  # helper function for data preparation
  def generate_experiment_data(data):
    experiment_data = pd.DataFrame({
      'instrument': list(data.columns) * len(data.index),
      'date': np.repeat(data.index, len(data.columns)),
      'factor': data.values.flatten()
    })

    experiment_data.set_index(['date', 'instrument'], inplace=True)

    return experiment_data
  
  experiment_data = generate_experiment_data(factor_data)

  # price should be shifted by 1 day to align with factor data
  # (factor data -> make decision -> next day price)
  price_data_shifted = price_data.shift(-1)

  # generating garphs
  graph_data = al.utils.get_clean_factor_and_forward_returns(experiment_data, price_data_shifted, quantiles=quantiles, max_loss=max_loss)
  al.tears.create_full_tear_sheet(graph_data)


def create_factor_graphs_v2(all_data, factor_data, factor_name, trade_interval='w', transaction_cost=0):

    # data preparation: stacking
    factor_df_stacked = factor_data.stack().reset_index()
    factor_df_stacked.columns = ['Date', 'Ticker', factor_name]

    # merge before applying function
    merged_data = df_factor_merge(all_data, factor_df_stacked, factor_name)
    cal_icir(merged_data, factor_name, trade_interval=trade_interval)
    group_backtest(merged_data, factor_name, trade_interval=trade_interval, transaction_cost=transaction_cost)


def z_score(data, f_n):
    """
    :param f_n: factor name
    :param data: factor df with columns like [Ticker, Date, factor_name]
    :return: df: factor after z score standardization
    """
    Ticker = 'Ticker'
    Date = 'Date'
    factor_name = f_n
    # cal mean of every cross-section
    c_mean = data.groupby(Date)[factor_name].mean()
    c_mean.name = 'mean'
    # cal std of every cross-section
    c_std = data.groupby(Date)[factor_name].std()
    c_std.name = 'std'
    # merge data
    df = pd.merge(data, c_mean, left_on=Date, right_index=True)
    df = pd.merge(df, c_std, left_on=Date, right_index=True)

    df[factor_name] = (df[factor_name] - df['mean']) / df['std']
    return df.drop(['mean', 'std'], axis=1).reset_index(drop=True)


def filter_extreme_3sigma(data):
    dt_up = data.mean() + 3 * data.std()
    dt_down = data.mean() - 3 * data.std()
    return data.clip(dt_down, dt_up)  # 超出上下限的值，赋值为上下限


def factor_washing(factor_df, factor_name, window_len=None):
    """
    :param factor_df: a dataframe, columns ['Ticker', 'Date', factor_name]
    :param factor_name: the name of the factor you want to test
    :param window_len: window size to aggregate the factor if needed
    :return: dataframe of the factor after data washing
    """

    # erase the extreme values
    factor_df[factor_name] = factor_df.groupby('Ticker')[factor_name].apply(
        filter_extreme_3sigma).reset_index(level=0, drop=True)

    # cal mean value in the window
    if window_len is not None:
        factor_df[factor_name] = factor_df.groupby('Ticker')[factor_name].rolling(
            window_len, min_periods=int(window_len / 2)).mean().reset_index(level=0, drop=True)

    # about the nan
    # if there are continuous 5 days or more with NULL, just drop
    # otherwise use ffill()
    factor_df['null_counts'] = factor_df[factor_name].isnull()
    factor_df['null_counts'] = factor_df['null_counts'].rolling(5).sum()
    factor_df = factor_df[factor_df['null_counts'] < 5]
    factor_df = factor_df[factor_df['null_counts'] < 5].drop(['null_counts'], axis=1)
    factor_df[factor_name] = factor_df.groupby('Ticker')[factor_name].apply(
        lambda x: x.ffill()).reset_index(level=0, drop=True)

    # z-score standardization
    factor_df = z_score(factor_df, factor_name)

    return factor_df.reset_index(drop=True)


def df_factor_merge(s_data, f_data, factor_name):
    """
    :param s_data: the data of stocks, actually only adj close is utilized
    :param f_data: the data of the factors
    :param factor_name: factor name
    :return:
    """
    m_d = pd.merge(s_data, f_data, on=['Ticker', 'Date'], how='right')

    # about the nan
    # if there are continuous 5 days or more with NULL, just drop
    # otherwise use ffill()
    m_d['null_counts'] = m_d[factor_name].isnull()
    m_d['null_counts'] = m_d['null_counts'].rolling(5).sum()
    m_d = m_d[m_d['null_counts'] < 5]
    m_d = m_d[m_d['null_counts'] < 5].drop(['null_counts'], axis=1)
    m_d[factor_name] = m_d.groupby('Ticker')[factor_name].apply(
        lambda x: x.ffill()).reset_index(level=0, drop=True)
    return m_d.dropna().reset_index(drop=True)


def cal_icir(data, factor_name, trade_interval='w'):
    """
    :param factor_name: factor name
    :param data: dataframe contains both stocks price and factor data
    :param trade_interval: the frequency we trade, choose in 'd'(daily), 'w'(weekly), 'm'(monthly)
    :return: a list contains the sequence of ic
    """
    data = data.copy()

    interval_dict = {'w': 5, 'd': 1, 'm': 20}
    lag_days = interval_dict[trade_interval]
    # calculate next term's return
    data['return'] = data['Adj Close'].shift(-lag_days) / data['Adj Open'].shift(-1) - 1

    # find the date we would trade
    if trade_interval != 'd':
        # set group frequency according to trade_interval
        freq = 'W' if trade_interval == 'w' else 'M'

        # Group by date and get the first trading day of each group
        data['temp_date'] = data['Date']
        trade_dates = data.groupby([pd.Grouper(key='temp_date', freq=freq), 'Ticker']).first()
        trade_dates = trade_dates.reset_index(drop=True)

        # Only the data of the trading day is retained
        data = data[data['Date'].isin(trade_dates['Date'].unique())]
    else:
        pass

    # calculate cross-section rank correlation coefficient
    ic_dict = {'Date': [], 'IC': []}
    ic_li = []
    for date, group in data.groupby('Date'):
        temp_ic = np.corrcoef(group[factor_name].rank(), group['return'].rank())[0, 1]
        ic_li.append(temp_ic)

        ic_dict['Date'].append(date)
        ic_dict['IC'].append(temp_ic)

    ic = np.nanmean(ic_li)
    ir = ic / np.nanstd(ic_li)
    print("the factor's ic is", ic)
    print("the factor's ir is", ir)
    print('\n')
    ic_li = np.array(ic_li)
    cum_ic = np.cumsum(ic_li)
    fig, ax1 = plt.subplots(figsize=(8, 4))

    # draw IC sequence bar chart
    ax1.bar(data['Date'].unique(), ic_li, color='blue', label='IC')
    ax1.set_xlabel('date')
    ax1.set_ylabel('IC', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')

    # the second Y axis
    ax2 = ax1.twinx()

    # draw accumulated IC line chart
    ax2.plot(data['Date'].unique(), cum_ic, color='orange', label='cum_IC')
    ax2.set_ylabel('cum_IC', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.legend(loc='upper right')

    # set title
    plt.title('IC & cum_IC')

    # show the figure
    plt.show()

    ic_df = pd.DataFrame(ic_dict)
    ic_df['Date'] = pd.to_datetime(ic_df['Date'])

    return ic_df


def factor_orthogonalization(data, method, factor_list):
    """
    :param data: dataframe contains factor data
    :param factor_list: A list of factor names
    :param method: the method to implement orthogonalization, 'pca' or 'gram_schmidt'
    :return: Factor data after orthogonalization
    """
    df = data.copy()

    if method == 'pca':
        # Cycle by date, PCA orthogonalization is performed for each section
        for date, group in df.groupby('Date'):
            # Extract factor data
            X = group[factor_list].values

            # standardization
            X = (X - X.mean(axis=0)) / X.std(axis=0)

            # Calculate the covariance matrix
            M = np.cov(X.T)

            # Calculate eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(M)

            # Construct orthogonal matrices
            eigenvectors = eigenvectors.real
            norms = np.linalg.norm(eigenvectors, axis=0)
            eigenvectors = eigenvectors / norms

            # Orthogonalization
            orthogonal_factors = X.dot(eigenvectors)

            # Update the data
            for i, factor in enumerate(factor_list):
                df.loc[group.index, f'{factor}_ortho'] = orthogonal_factors[:, i]

    elif method == 'gram_schmidt':
        # Cycle by date, Gram-Schmidt orthogonalization is performed on each section
        for date, group in df.groupby('Date'):
            # Extract factor data
            X = group[factor_list].values

            # standardization
            X = (X - X.mean(axis=0)) / X.std(axis=0)

            # Gram-Schmidt orthogonalization
            Q = np.zeros_like(X)
            Q[:, 0] = X[:, 0]  # The first factor remains the same

            for i in range(1, X.shape[1]):
                # Calculate the projection
                proj = np.zeros_like(X[:, 0])
                for j in range(i):
                    proj += (np.dot(X[:, i], Q[:, j]) / np.dot(Q[:, j], Q[:, j])) * Q[:, j]

                # Orthogonalization
                Q[:, i] = X[:, i] - proj

            # Standardized orthogonal basis
            Q = Q / np.linalg.norm(Q, axis=0)

            # Update the data
            for i, factor in enumerate(factor_list):
                df.loc[group.index, f'{factor}_ortho'] = Q[:, i]

    # The correlation matrix before and after orthogonalization was calculated
    original_corr = data[factor_list].corr()
    ortho_factors = [f'{f}_ortho' for f in factor_list]
    ortho_corr = df[ortho_factors].corr()

    # Draw a heat map of the correlation before and after orthogonalization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    sns.heatmap(original_corr,
                annot=True,
                cmap='RdYlBu',
                center=0,
                fmt='.2f',
                square=True,
                ax=ax1)
    ax1.set_title('Correlation matrix before orthogonalization')

    sns.heatmap(ortho_corr,
                annot=True,
                cmap='RdYlBu',
                center=0,
                fmt='.2f',
                square=True,
                ax=ax2)
    ax2.set_title('Correlation matrix after orthogonalization')

    plt.tight_layout()
    plt.show()

    return df


def group_backtest(m_data, factor_name, group_num=5, trade_interval='w', transaction_cost=0, direction='positive', index=None):
    """
        :param m_data: dataframe contains both stocks price and factor data
        :param factor_name: factor name
        :param group_num: number of groups, 5 by default
        :param trade_interval: the frequency we trade, choose in 'd'(daily), 'w'(weekly), 'm'(monthly)
        :param transaction_cost: The cost of a single side is 0 by default, if set to 0.0015, it means 0.15% of the
         transaction cost
        :param direction: the direction of the factor, 'positive' or 'negative'
        :param index: SP500 index data, optional
        :return: Returns the cumulative yield data for each group
        """
    data = m_data.copy()

    # Set the transaction interval
    interval_dict = {'w': 5, 'd': 1, 'm': 20}
    lag_days = interval_dict[trade_interval]

    # Calculate future yields
    # Buy and sell each bear the transaction cost
    data['return'] = data.groupby('Ticker').apply(
        lambda x: x['Adj Close'].shift(-lag_days) / x['Adj Close'] - 1).reset_index(level=0, drop=True)

    data.dropna(inplace=True)

    # Get the date of the transaction
    if trade_interval != 'd':
        # set group frequency according to trade_interval
        freq = 'W' if trade_interval == 'w' else 'M'

        # Group by date and get the first trading day of each group
        data['temp_date'] = data['Date']
        trade_dates = data.groupby([pd.Grouper(key='temp_date', freq=freq), 'Ticker'])['temp_date'].last()
        trade_dates = trade_dates.reset_index(drop=True).unique()

        # Only the data of the trading day is retained
        data = data[data['Date'].isin(trade_dates)]
    else:
        pass

    # Initialize the yield record
    returns_dict = {f'group_{i + 1}': [] for i in range(group_num)}
    returns_dict['long_short'] = []

    # Group backtesting by date
    for date, group in data.groupby('Date'):
        # Group by factor value
        group['group'] = pd.qcut(group[factor_name], group_num, labels=[f'group_{i + 1}' for i in range(group_num)])

        # Calculate the yield of each group
        group_returns = group.groupby('group', observed=False)['return'].mean()

        # Record the yields of each group
        for g in returns_dict.keys():
            if g == 'long_short':
                if direction == 'positive':
                    temp = group_returns[f'group_{group_num}'] - group_returns['group_1']
                else:
                    temp = group_returns['group_1'] - group_returns[f'group_{group_num}']
                returns_dict[g].append(temp - (temp + 1 - transaction_cost) * transaction_cost - transaction_cost)
            else:
                temp =group_returns[g]
                returns_dict[g].append(temp - (temp + 1 - transaction_cost) * transaction_cost - transaction_cost)

    # transform to DataFrame
    returns_df = pd.DataFrame(returns_dict, index=data['Date'].unique())

    # Calculate the cumulative rate of return
    cum_returns = (1 + returns_df).cumprod()

    # Calculate statistical metrics
    annual_returns = (returns_df.mean() * 252 / lag_days).round(4)
    annual_volatility = (returns_df.std() * np.sqrt(252 / lag_days)).round(4)
    sharpe_ratio = (annual_returns / annual_volatility).round(4)
    max_drawdown = ((cum_returns / cum_returns.cummax() - 1).min()).round(4)

    # Plot the cumulative rate of return
    fig, ax1 = plt.subplots(figsize=(8, 4))

    # Plotting the Grouped Yield Curve (Left Y-axis)
    for col in cum_returns.columns:
        if col != 'long_short':
            ax1.plot(cum_returns.index, cum_returns[col], label=col)

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Return (Groups)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # Create a second Y-axis
    ax2 = ax1.twinx()

    # Plotting the long-short yield curve (right Y-axis)
    ax2.plot(cum_returns.index, cum_returns['long_short'],
             label='long_short', color='gray', linestyle='--', linewidth=2)
    ax2.set_ylabel('Cumulative Return (Long-Short)', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    # Legends for merging two axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title('Cumulative Returns of Different Groups ' + factor_name)
    plt.grid(True)
    plt.show()

    # Print statistical metrics
    print("\n=== Backtesting statistical metrics ===")
    print(f"transaction_cost: {transaction_cost * 100:.2f}%(single side)")
    print(f"Annualized return:\n{annual_returns}")
    print(f"\nAnnualized volatility:\n{annual_volatility}")
    print(f"\nSharpe ratio:\n{sharpe_ratio}")
    print(f"\nmax drawback:\n{max_drawdown}")

    # Plot the alpha of the long group
    if index is not None:
        # 计算指数收益率
        index_returns = index.set_index('Date')['return'].rolling(lag_days).apply(lambda x: math.prod(x + 1) - 1)
        index_returns = index_returns.shift(-lag_days)
        index_returns = index_returns[data['Date'].unique()]

        # 获取多头组收益（根据因子方向选择最高或最低组）
        if direction == 'positive':
            long_returns = returns_df[f'group_{group_num}']
        else:
            long_returns = returns_df['group_1']

        # 计算超额收益
        alpha_returns = long_returns - index_returns
        cum_alpha = (1 + alpha_returns).cumprod()

        # 绘制超额收益图
        plt.figure(figsize=(8, 4))
        # plt.plot(cum_alpha.index, (1 + long_returns).cumprod(), label='long group', color='orange')
        plt.plot(cum_alpha.index, (1 + long_returns).cumprod(), label='Portfolio', color='orange')
        plt.plot(cum_alpha.index, (1 + index_returns).cumprod(), label='S&P 500', color='green')
        plt.plot(cum_alpha.index, cum_alpha, label='Cumulative Alpha', color='blue')
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.3)
        # plt.xlabel('Date')
        plt.ylabel('Cumulative Alpha Return')
        plt.title('Portfolio vs Benchmark')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 打印超额收益统计指标
        annual_long = (long_returns.mean() * 252 / lag_days).round(4)
        long_volatility = (long_returns.std() * np.sqrt(252 / lag_days)).round(4)
        long_sharpe = (annual_long / long_volatility).round(4)
        long_max_drawdown = np.round((((1 + long_returns).cumprod() / (1 + long_returns).cumprod().cummax() - 1).min()), 4)

        print("\n=== Long Group statistical metrics ===")
        print(f"Annualized return: {annual_long:.4f}")
        print(f"Long volatility: {long_volatility:.4f}")
        print(f"Long sharpe: {long_sharpe:.4f}")
        print(f"Long max drawdown: {long_max_drawdown:.4f}")

        # 打印超额收益统计指标
        annual_alpha = (alpha_returns.mean() * 252 / lag_days).round(4)
        alpha_volatility = (alpha_returns.std() * np.sqrt(252 / lag_days)).round(4)
        alpha_sharpe = (annual_alpha / alpha_volatility).round(4)
        alpha_max_drawdown = np.round(((cum_alpha / cum_alpha.cummax() - 1).min()), 4)

        print("\n=== Alpha statistical metrics ===")
        print(f"Annualized alpha: {annual_alpha:.4f}")
        print(f"Alpha volatility: {alpha_volatility:.4f}")
        print(f"Alpha sharpe: {alpha_sharpe:.4f}")
        print(f"Alpha max drawdown: {alpha_max_drawdown:.4f}")

        return {'annual_return': annual_long, 'sharpe': long_sharpe}

    return cum_returns
    


def neutralization(data, factor_name, neutral_list=None):
    """
    :param data: dataframe contains both stocks price and factor data
    :param factor_name: Neutral factor names are required
    :param neutral_list: Neutral processing list, optional 'industry' and 'value'
    :return: Neutralized factor value
    """
    if neutral_list is None:
        neutral_list = ['industry', 'value']
    df = data.copy()

    # Cycle by date, neutralizing each section
    neutral_factor = []

    for date, group in df.groupby('Date'):
        # Prepare the Dependent Variable (Factor Value)
        y = group[factor_name].values

        # Prepare independent variables (market capitalization and sector dummy variables)
        X_list = []

        if 'value' in neutral_list:
            # Take the logarithm of the market capitalization
            log_value = np.log(group['value']).values.reshape(-1, 1)
            X_list.append(log_value)

        if 'industry' in neutral_list:
            # Generate industry dummy variables
            industry_dummies = pd.get_dummies(group['Industry'])
            # Remove an industry to avoid multicollinearity
            industry_dummies = industry_dummies.iloc[:, :-1]
            X_list.append(industry_dummies)

        # Merge arguments
        if X_list:
            X = np.hstack(X_list)

            # Add a constant term
            X = sm.add_constant(X)

            # Perform a regression
            model = sm.OLS(y, X)
            results = model.fit()

            # Calculate the residuals as the neutralized factor values
            neutral_factor.extend(results.resid)
        else:
            neutral_factor.extend(y)

    new_name = factor_name + '_neutral'
    df[new_name] = neutral_factor

    return df


def plot_factor_correlation_matrix(data, factor_list):
    """
    :param data: dataframe contains factor data
    :param factor_list: A list of factor names
    """
    # Calculate the average cross-section correlation matrix
    corr_matrix = pd.DataFrame(index=factor_list, columns=factor_list)

    for f1 in factor_list:
        for f2 in factor_list:
            corr_series = []
            for date, group in data.groupby('Date'):
                corr = group[f1].corr(group[f2])
                corr_series.append(corr)
            corr_matrix.loc[f1, f2] = np.mean(corr_series)

    # Draw a heat map
    plt.figure(figsize=(8, 4))
    sns.heatmap(corr_matrix,
                annot=True,
                cmap='RdYlBu',
                center=0,
                fmt='.2f',
                square=True)
    plt.title('Factor correlation matrix')
    plt.show()

    return corr_matrix

def factor_weighted(factor_df, factor_list, method='ic_weight', lookback = 20, half_life=6):
    '''
    使用factor_weighted_combine_rolling方法实现的因子加权组合函数, 自动计算ic值, 无需提供ic_df

    参数:
    - factor_df: DataFrame, 包含 ['Date', 'Ticker', 'factor1', 'factor2', ...]
    - factor_list: 需要组合的因子列表, list of str
    - method: 加权方法 ['equal', 'ic_weight', 'exp_ic_weight', 'vol_adj']
    - lookback: 回溯窗口长度
    - half_life: 指数加权的半衰期
    '''
    factor_df_copy = factor_df.copy()

    ic_df = None
    for factor in factor_list:
        # calculating current facotr's ic values and prepare data
        ic = cal_icir(factor_df_copy, factor, trade_interval='d')
        ic.rename(columns={'IC': f'{factor}_ic'}, inplace=True)

        if ic_df is None:
            ic_df = ic
        else:
            ic_df = pd.merge(ic_df, ic, on='Date', how='inner')
        
    # calculating weighted factor values
    factor_weighted = factor_weighted_combine_rolling(factor_df_copy, ic_df, factor_list, method=method, lookback=lookback, half_life=half_life)
    return factor_weighted


def factor_weighted_combine_rolling(factor_df, ic_df, factor_list, method='ic_weight',
                                    lookback=20, half_life=6):
    """
    使用rolling方法实现的因子加权组合函数

    参数:
    - factor_df: DataFrame, 包含 ['Date', 'Ticker', 'factor1', 'factor2', ...]
    - ic_df: DataFrame, 包含 ['Date', 'factor1_ic', 'factor2_ic', ...]
    - factor_list: 需要组合的因子列表
    - method: 加权方法 ['equal', 'ic_weight', 'exp_ic_weight', 'vol_adj']
    - lookback: 回溯窗口长度
    - half_life: 指数加权的半衰期
    """
    # 1. 数据准备
    factor_data = factor_df.copy()
    ic_data = ic_df.copy()
    ic_data.set_index('Date', inplace=True)

    # 对每个因子进行截面标准化
    for factor in factor_list:
        factor_data[factor] = factor_data.groupby('Date')[factor].transform(
            lambda x: (x - x.mean()) / x.std()
        )

    # 2. 计算因子权重
    if method == 'equal':
        # 等权重
        weights = pd.DataFrame(
            1 / len(factor_list),
            index=ic_data.index,
            columns=factor_list
        )

    elif method == 'ic_weight':
        # 简单IC加权
        ic_abs = ic_data[[f'{f}_ic' for f in factor_list]].abs()
        weights = ic_abs.rolling(window=lookback, min_periods=1).mean()
        # 重命名列以匹配因子列表
        weights.columns = factor_list
        # 标准化权重使其和为1
        weights = weights.div(weights.sum(axis=1), axis=0)

    elif method == 'exp_ic_weight':
        # 指数加权IC
        ic_abs = ic_data[[f'{f}_ic' for f in factor_list]].abs()
        weights = ic_abs.ewm(
            halflife=half_life,
            min_periods=1,
            adjust=False
        ).mean()
        # 重命名列以匹配因子列表
        weights.columns = factor_list
        # 标准化权重使其和为1
        weights = weights.div(weights.sum(axis=1), axis=0)

    elif method == 'vol_adj':
        # 波动率调整加权
        # 计算每个因子的滚动波动率
        factor_vol = pd.DataFrame(index=ic_data.index, columns=factor_list)

        for factor in factor_list:
            daily_std = factor_data.groupby('Date')[factor].std()
            factor_vol[factor] = daily_std.rolling(window=lookback, min_periods=1).mean()

        # 使用波动率倒数作为权重
        weights = 1 / factor_vol
        weights = weights.div(weights.sum(axis=1), axis=0)

    # 3. 计算加权因子
    # 创建结果DataFrame
    weighted_factor = pd.DataFrame(index=factor_data.index)
    weighted_factor['Date'] = factor_data['Date']
    weighted_factor['Ticker'] = factor_data['Ticker']

    # 对每个日期应用权重
    for date in factor_data['Date'].unique():
        if date in weights.index:
            date_weights = weights.loc[date]
            date_factors = factor_data[factor_data['Date'] == date][factor_list]

            weighted_factor.loc[factor_data['Date'] == date, 'weighted_factor'] = \
                (date_factors * date_weights).sum(axis=1)

    return weighted_factor

if __name__ == '__main__':

    # testing
    path = 'factors'
    all_data = pd.read_pickle('all_data.pkl')
    stacked_factors = ['beta_daily.pkl', 'beta_opening_hours.pkl', 'beta_closing_hours.pkl', 'rsi.pkl']
    factor_str_lst = os.listdir(path)
    factor_str_lst = list(set(factor_str_lst) - set(stacked_factors))

    # merge all factors in all_data
    for factor in factor_str_lst:
        if factor.endswith('.pkl'):
            factor_name = factor.split('.')[0]
            factor_data = pd.read_pickle(f'{path}/{factor}')
            # stack factor data to match the format of all_data
            factor_data = factor_data.stack().reset_index()
            factor_data.columns = ['Date', 'Ticker', factor_name]
            factor_washing(factor_data, factor_name, window_len=20)
            try:
                all_data = df_factor_merge(all_data, factor_data, factor_name=factor_name)
            except Exception as e:
                print(e)
                print(factor_name)

    # merging
    beta_factors = ['beta_daily.pkl', 'beta_opening_hours.pkl', 'beta_closing_hours.pkl']
    for factor in beta_factors:
        factor_name = factor.split('.')[0]
        factor_data = pd.read_pickle(f'{path}/{factor}')
        factor_washing(factor_data, factor_name, window_len=20)
        all_data = df_factor_merge(all_data, factor_data, factor_name=factor_name)

    all_data_temp = all_data.copy()
    factor_name1 = 'high_freq_down_vol_ratio'
    factor_name2 = 'beta_daily'
    ic_1 = cal_icir(all_data_temp, factor_name1, trade_interval='d')
    ic_2 = cal_icir(all_data_temp, factor_name2, trade_interval='d')

    ic_1.rename(columns={'IC': f'{factor_name1}_ic'}, inplace=True)
    ic_2.rename(columns={'IC': f'{factor_name2}_ic'}, inplace=True)

    ic_df = pd.merge(ic_1, ic_2, on='Date', how='inner')

    factor_weighted = factor_weighted_combine_rolling(all_data_temp, ic_df, [factor_name1, factor_name2], method='ic_weight', lookback=20)