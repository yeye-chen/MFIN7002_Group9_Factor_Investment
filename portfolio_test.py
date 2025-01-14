from squeezing_noise import *

# 禁用所有警告
warnings.filterwarnings('ignore')


# 测试函数
def test_on_factor(data, long_group, q, alpha, test_window=240, forward_window=5, ll=None):
    """
    基于因子选股结果进行组合优化测试

    参数:
    - data: 收益率数据 (DataFrame)
    - long_group: 每期选出的股票代码字典 {date: [stock_codes]}
    - q, alpha: 降噪参数
    - test_window: 计算协方差矩阵的窗口长度
    - forward_window: 前瞻验证窗口长度
    - ll: 需要测试的策略列表
    """
    if ll is None:
        ll = ['denoised', 'original', 'equal_weight', 'cluster', 'pca', 'momentum']

    # 初始化结果字典
    results = {}
    for model in ll:
        results[model] = {'returns': [], 'd_returns': [], 'volatility': [], 'sharpe': [], 'sequence': []}

    # 获取所有调仓日期
    time_points = sorted(long_group.keys())

    daily_returns = {}

    for t in time_points:
        # 获取当期选中的股票
        current_stocks = long_group[t]

        # 确保有足够的历史数据
        if t < data.index[test_window]:
            continue

        # 获取历史数据（仅包含当期选中的股票）
        hist_data = data[current_stocks]
        hist_data = hist_data[hist_data.index <= t].tail(test_window).fillna(0)

        hist_data = drop_consecutive_zeros(hist_data)

        current_stocks = hist_data.columns

        # 获取未来数据
        future_data = data[current_stocks]
        future_data = future_data[future_data.index > t].head(forward_window)

        # 如果数据不足，跳过当期
        if len(hist_data) < test_window or len(future_data) < forward_window:
            continue

        # 计算降噪相关矩阵
        squeezed_corr = squeeze_correlation_matrix(returns=hist_data, q=q, alpha=alpha)

        final_s = []

        # 1. 使用降噪方法
        if 'denoised' in ll:
            denoised_weights = portfolio_optimization_with_denoising(
                returns=hist_data,
                method='mean_variance',
                alpha=alpha,
                q=q,
                max_weight=0.3
            )
            final_s.append(('denoised', denoised_weights))

        # 2. 使用原始方法
        if 'original' in ll:
            original_weights = portfolio_optimization_without_denoising(
                returns=hist_data,
                method='mean_variance',
                max_weight=0.3
            )
            final_s.append(('original', original_weights))

        # 3. 使用等权重
        if 'equal_weight' in ll:
            equal_weights = np.ones(len(current_stocks)) / len(current_stocks)
            final_s.append(('equal_weight', equal_weights))

        # 4. 使用机器学习方法
        if any(method in ll for method in ['cluster', 'sparse', 'pca', 'momentum']):
            ml_portfolios = build_ml_portfolios(hist_data, squeezed_corr, lookback=min(60, test_window))

            if 'cluster' in ll:
                final_s.append(('cluster', ml_portfolios['cluster']))
            if 'sparse' in ll:
                final_s.append(('sparse', ml_portfolios['sparse']))
            if 'pca' in ll:
                final_s.append(('pca', ml_portfolios['pca']))
            if 'momentum' in ll:
                final_s.append(('momentum', ml_portfolios['momentum']))

        # 计算每个策略的表现
        for method, weights in final_s:
            # 计算组合收益
            portfolio_returns = (future_data * weights).sum(axis=1)

            # 计算统计指标
            ret = math.prod(portfolio_returns + 1)
            d_ret = np.mean(portfolio_returns)
            vol = portfolio_returns.std()
            sharpe = d_ret / vol if vol != 0 else 0

            # 存储结果
            results[method]['returns'].append(ret)
            results[method]['d_returns'].append(d_ret)
            results[method]['volatility'].append(vol)
            results[method]['sharpe'].append(sharpe)
            results[method]['sequence'] = results[method]['sequence'] + portfolio_returns.to_list()

            if method == 'pca':
                daily_returns[t] = portfolio_returns

    # 汇总结果
    summary = {}
    for method in ll:
        summary[method] = {
            'avg_daily_return': np.mean(results[method]['d_returns']),
            'cum_return': math.prod(results[method]['returns']),
            'avg_volatility': np.mean(results[method]['volatility']),
            'total_sharpe': np.mean(results[method]['sequence']) / np.std(results[method]['sequence']),
            'w_avg_sharpe': np.mean(results[method]['sharpe']),
            'return_std': np.std(results[method]['returns']),
            'w_sharpe_std': np.std(results[method]['sharpe'])
        }

    # 打印结果
    print("\n=== 因子组合优化结果 ===")
    for method in summary.keys():
        print(f"\n{method}策略:")
        for metric, value in summary[method].items():
            print(f"{metric}: {value:.4f}")

    # 进行统计检验
    methods = list(summary.keys())
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            method1, method2 = methods[i], methods[j]
            t_stat, p_value = stats.ttest_ind(
                results[method1]['sharpe'],
                results[method2]['sharpe']
            )
            print(f"\n{method1} vs {method2}夏普比率差异显著性检验:")
            print(f"t统计量: {t_stat:.4f}")
            print(f"p值: {p_value:.4f}")

    return summary, results, daily_returns


# 获取股票收益数据
returns_data = pd.read_pickle('returns_matrix.pkl')

# 获取因子数据
factor_data = pd.read_pickle('best_factor.pkl')

# 按因子得到每周多头组
freq = 'W'

factor_data['temp_date'] = factor_data['Date']
trade_dates = factor_data.groupby([pd.Grouper(key='temp_date', freq=freq), 'Ticker'])['temp_date'].last()
trade_dates = trade_dates.reset_index(drop=True).unique()

tte = factor_data[factor_data['Date'].isin(trade_dates)]
tte['group'] = tte.groupby('Date').apply(lambda x: pd.qcut(x['best_factor'], 10, labels=[i+1 for i in range(10)])
                                         ).reset_index(level=0, drop=True)
l_g_all = tte[tte['group'] == 10]

time_point = l_g_all.Date.sort_values().unique()

group_dict = {}
for t in time_point[:-1]:
    group_dict[t] = list(l_g_all[l_g_all['Date'] == t]['Ticker'])

# 使用优化器计算每周投资组合的权重
summary, result, daily_r = test_on_factor(returns_data, group_dict, q=6, alpha=1.2, ll=['pca'])

sp500_data = pd.read_pickle('data/sp_500.pkl')

former_d = None
former_l = None
s_l = []
for d, l in daily_r.items():
    if former_d is None:
        pass
    else:
        print(d)
        temp = sp500_data[(sp500_data['Date'] > former_d) & (sp500_data['Date'] <= d)]
        valid_len = temp.shape[0]
        s_l.append(pd.Series(data=former_l[:(valid_len)], index=temp['Date']))
    former_l = l
    former_d = d
f_temp = sp500_data[(sp500_data['Date'] > d) & (sp500_data['Date'] <= pd.to_datetime('2024-12-06'))]
f_valid_len = f_temp.shape[0]
s_l.append(pd.Series(data=l[:(f_valid_len)], index=f_temp['Date']))
see = pd.concat(s_l)

sp500_data_i = sp500_data[sp500_data['Date'].isin(see.index)].copy()
sp500_data_i.set_index('Date', inplace=True)
cum_alpha = (see - sp500_data_i['return'] + 1).cumprod()
plt.figure(figsize=(12, 6))
plt.plot(see.index, (1 + see).cumprod(), label='long group', color='orange')
plt.plot(see.index, (1 + sp500_data_i['return']).cumprod(), label='S&P 500', color='green')
plt.plot(see.index, cum_alpha, label='Cumulative Alpha', color='blue')
plt.axhline(y=1, color='r', linestyle='--', alpha=0.3)
plt.xlabel('Date')
plt.ylabel('Cumulative Alpha Return')
plt.title('Long Portfolio Alpha vs Index ' + 'best_factor')
plt.legend()
plt.grid(True)
plt.show()