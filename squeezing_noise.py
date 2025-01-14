import math
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh
import pandas as pd
import random
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.covariance import GraphicalLasso
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import warnings

# 禁用所有警告
warnings.filterwarnings('ignore')


def squeeze_correlation_matrix(returns, q=None, alpha=2):
    """
    Gerber的Squeeze方法实现

    参数:
    - returns: 收益率数据
    - q: T/N比率，T是样本数，N是资产数
    - alpha: 挤压参数，控制变换的强度
    """
    # 计算相关系数矩阵
    corr = returns.corr()
    N = len(corr)
    T = len(returns)

    # 如果未指定q，则计算实际比率
    if q is None:
        q = T / N

    # 1. 特征值分解
    eigenvals, eigenvecs = eigh(corr)
    # 降序排列
    idx = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]

    # 2. 计算Marcenko-Pastur分布的参数
    lambda_plus = (1 + 1 / np.sqrt(q)) ** 2  # MP分布上界
    lambda_minus = (1 - 1 / np.sqrt(q)) ** 2  # MP分布下界

    # 3. 应用Squeeze变换
    def squeeze_function(lambda_val):
        """
        改进的Squeeze变换函数：
        - 对超出MP分布边界的特征值进行更强的压缩
        - 对在边界内的特征值进行温和处理
        """
        # if lambda_val > lambda_plus:
        #     # 对大于上界的特征值进行更强的压缩
        #     return 1 + (lambda_val - 1) / (alpha * 2 * (1 + (lambda_val - 1)))
        # elif lambda_val < lambda_minus:
        #     # 对小于下界的特征值进行更强的压缩
        #     return 1 - (1 - lambda_val) / (alpha * 2 * (1 + (1 - lambda_val)))
        if lambda_val < lambda_minus:
            # 对小于下界的特征值进行更强的压缩
            return 1 - (1 - lambda_val) / (alpha * 2 * (1 + (1 - lambda_val)))
        else:
            # 对在MP分布边界内的特征值进行温和处理
            if lambda_val > 1:
                return 1 + (lambda_val - 1) / (alpha * (1 + (lambda_val - 1)))
            else:
                return 1 - (1 - lambda_val) / (alpha * (1 + (1 - lambda_val)))

    # 4. 对特征值进行变换
    squeezed_eigenvals = np.array([squeeze_function(ev) for ev in eigenvals])

    # 5. 重构相关矩阵
    squeezed_corr = np.zeros_like(corr)
    for i in range(N):
        squeezed_corr += squeezed_eigenvals[i] * np.outer(eigenvecs[:, i], eigenvecs[:, i])

    # 6. 确保对角线为1
    np.fill_diagonal(squeezed_corr, 1)

    return squeezed_corr


def portfolio_optimization_with_denoising(returns, q=None, method='risk_parity', alpha=2, max_weight=0.1):
    """
    结合降噪和传统优化方法的组合优化器

    参数:
    - returns: 收益率数据
    - method: 优化方法 ('mean_variance', 'min_variance', 'risk_parity')
    - alpha: squeeze降噪参数
    - max_weight: 最大权重限制
    """
    # 1. 计算基础统计量
    mean_returns = returns.mean()
    std_returns = returns.std()

    # 2. 对相关矩阵进行降噪
    squeezed_corr = squeeze_correlation_matrix(returns, q, alpha=alpha)

    # 3. 重构协方差矩阵
    denoised_cov = np.outer(std_returns, std_returns) * squeezed_corr

    # 4. 根据选择的方法进行优化
    n_assets = len(returns.columns)

    if method == 'mean_variance':
        # 均值方差优化
        def objective(weights):
            port_return = np.sum(weights * mean_returns)
            port_std = np.sqrt(np.dot(weights.T, np.dot(denoised_cov, weights)))
            return -port_return / port_std  # 最大化夏普比率

    elif method == 'min_variance':
        # 最小方差优化
        def objective(weights):
            return np.sqrt(np.dot(weights.T, np.dot(denoised_cov, weights)))

    elif method == 'risk_parity':
        # 风险平价优化
        def objective(weights):
            port_std = np.sqrt(np.dot(weights.T, np.dot(denoised_cov, weights)))
            risk_contrib = weights * (np.dot(denoised_cov, weights)) / port_std
            target_risk = port_std / n_assets
            return np.sum((risk_contrib - target_risk) ** 2)

    # 5. 设置约束条件
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 权重和为1
        {'type': 'ineq', 'fun': lambda x: max_weight - x},  # 上限约束
        {'type': 'ineq', 'fun': lambda x: x - 0.01}  # 非负约束
    ]

    # 6. 求解优化问题
    from scipy.optimize import minimize
    result = minimize(
        objective,
        x0=np.ones(n_assets) / n_assets,  # 等权重初始值
        method='SLSQP',
        constraints=constraints,
        options={'maxiter': 1000}
    )

    # 清理权重：将非常小的负数设为0，并重新归一化
    weights = result.x
    weights[weights < 0] = 0  # 将负数设为0
    weights = weights / weights.sum()  # 重新归一化确保和为1
    return weights


def analyze_squeeze_effect(returns, alphas=[1.5, 2, 3]):
    """
    分析不同alpha值对特征值分布的影响
    """

    corr = returns.corr()
    eigenvals = eigh(corr)[0][::-1]

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(eigenvals) + 1), eigenvals, 'b-', label='Original')

    for alpha in alphas:
        squeezed_corr = squeeze_correlation_matrix(returns, alpha=alpha)
        squeezed_eigenvals = eigh(squeezed_corr)[0][::-1]
        plt.plot(range(1, len(squeezed_eigenvals) + 1),
                 squeezed_eigenvals, '--',
                 label=f'Squeezed (α={alpha})')

    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue')
    plt.legend()
    plt.title('Effect of Squeeze Transformation on Eigenvalues')
    plt.show()


def build_ml_portfolios(returns, squeezed_corr, lookback=60):
    """
    使用机器学习方法构建投资组合

    参数:
    - returns: 收益率数据
    - squeezed_corr: 降噪后的相关矩阵
    - lookback: 用于特征构建的历史窗口
    """
    n_assets = len(returns.columns)

    def cluster_based_allocation():
        """
        基于K-means聚类的配置方法
        将资产分组后，对每组进行等权重配置
        """
        n_clusters = min(5, n_assets // 3)  # 动态确定聚类数

        # 使用降噪后的相关矩阵作为特征
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(squeezed_corr)

        # 分配权重
        weights = np.zeros(n_assets)
        for i in range(n_clusters):
            cluster_assets = np.where(clusters == i)[0]
            cluster_weight = 1.0 / n_clusters
            weights[cluster_assets] = cluster_weight / len(cluster_assets)

        return weights

    def sparse_portfolio():
        """
        使用GraphicalLasso构建稀疏投资组合
        通过L1正则化实现自动特征选择
        """
        # 使用GraphicalLasso估计稀疏精度矩阵
        gl = GraphicalLasso(alpha=0.5)
        gl.fit(returns.iloc[-lookback:])

        # 使用精度矩阵的对角线元素作为权重基础
        weights = np.abs(np.diag(gl.precision_))
        weights = weights / np.sum(weights)

        return weights

    def pca_based_weights():
        """
        基于PCA的投资组合构建
        使用主成分权重作为投资组合权重
        """
        pca = PCA(n_components=0.95)  # 保留95%的方差
        pca.fit(returns.iloc[-lookback:])

        # 使用第一个主成分的绝对值作为权重
        weights = np.abs(pca.components_[0])
        weights = weights / np.sum(weights)

        return weights

    def momentum_based_weights():
        """
        基于动量的机器学习配置方法
        使用随机森林预测未来收益
        """
        # 构建特征
        X = []
        y = []
        for i in range(lookback, len(returns)):
            features = returns.iloc[i - lookback:i].mean()  # 使用历史均值作为特征
            target = returns.iloc[i]  # 使用下一期收益作为目标
            X.append(features)
            y.append(target)

        X = np.array(X)
        y = np.array(y)

        # 训练随机森林
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)

        # 使用最新数据预测
        latest_features = returns.iloc[-lookback:].mean()
        predicted_returns = rf.predict(latest_features.values.reshape(1, -1))[0]

        # 将预测转换为权重
        weights = np.maximum(predicted_returns, 0)  # 只取正预测值
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(n_assets) / n_assets

        return weights

    # 构建组合字典
    portfolios = {
        'cluster': cluster_based_allocation(),
        'sparse': sparse_portfolio(),
        'pca': pca_based_weights(),
        'momentum': momentum_based_weights()
    }

    return portfolios


def portfolio_optimization_without_denoising(returns, method='mean_variance', max_weight=0.1):
    """
    传统的投资组合优化方法（不进行降噪）

    参数:
    - returns: 收益率数据
    - method: 优化方法 ('mean_variance', 'min_variance', 'risk_parity')
    - max_weight: 最大权重限制
    """
    # 1. 计算基础统计量
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # 2. 根据选择的方法进行优化
    n_assets = len(returns.columns)

    if method == 'mean_variance':
        def objective(weights):
            port_return = np.sum(weights * mean_returns)
            port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -port_return / port_std  # 最大化夏普比率

    elif method == 'min_variance':
        def objective(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    elif method == 'risk_parity':
        def objective(weights):
            port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            risk_contrib = weights * (np.dot(cov_matrix, weights)) / port_std
            target_risk = port_std / n_assets
            return np.sum((risk_contrib - target_risk) ** 2)

    # 设置约束条件
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 权重和为1
        {'type': 'ineq', 'fun': lambda x: max_weight - x},  # 上限约束
        {'type': 'ineq', 'fun': lambda x: x - 0.01}  # 非负约束
    ]

    # 求解优化问题
    result = minimize(
        objective,
        x0=np.ones(n_assets) / n_assets,
        method='SLSQP',
        constraints=constraints,
        options={'maxiter': 1000}
    )

    # 清理权重：将非常小的负数设为0，并重新归一化
    weights = result.x
    weights[weights < 0] = 0  # 将负数设为0
    weights = weights / weights.sum()  # 重新归一化确保和为1
    return weights


def compare_optimization_methods(returns_data, q=None, alpha=2, test_window=24, forward_window=5,
                                 ll=None):
    """
    比较降噪和未降噪方法的表现

    参数:
    - returns_data: 收益率数据
    - test_window: 计算协方差矩阵的窗口长度
    - forward_window: 前瞻验证窗口长度
    """
    if ll is None:
        ll = ['denoised', 'original', 'equal_weight', 'sparse', 'pca', 'momentum']

    results = {}
    for model in ll:
        results[model] = {'returns': [], 'd_returns': [], 'volatility': [], 'sharpe': [], 'sequence': []}

    # 获取所有时间点
    time_points = returns_data.index[test_window:-forward_window:5]

    for t in time_points:
        # 获取历史数据
        hist_data = returns_data[returns_data.index <= t].tail(test_window)
        # 获取未来数据（用于验证）
        future_data = returns_data[returns_data.index > t].head(forward_window)

        final_s = []
        # 1. 使用降噪方法
        denoised_weights = portfolio_optimization_with_denoising(
            returns=hist_data,
            method='mean_variance',
            alpha=alpha,
            q=q,
            max_weight=0.3
        )
        final_s.append(('denoised', denoised_weights))

        if 'original' in ll:
            # 2. 使用原始方法
            original_weights = portfolio_optimization_without_denoising(
                returns=hist_data,
                method='mean_variance',
                max_weight=0.3
            )
            final_s.append(('original', original_weights))

        if 'equal_weight' in ll:
            # 3. 使用等权重
            equal_weights = np.ones(len(hist_data.columns)) / len(hist_data.columns)
            final_s.append(('equal_weight', equal_weights))

        # 4. 使用机器学习方法
        if any(method in ll for method in ['cluster', 'sparse', 'pca', 'momentum']):
            squeezed_corr = squeeze_correlation_matrix(hist_data, alpha=alpha, q=q)
            ml_portfolios = build_ml_portfolios(hist_data, squeezed_corr, lookback=min(60, test_window))

            if 'cluster' in ll:
                final_s.append(('cluster', ml_portfolios['cluster']))
            if 'sparse' in ll:
                final_s.append(('sparse', ml_portfolios['sparse']))
            if 'pca' in ll:
                final_s.append(('pca', ml_portfolios['pca']))
            if 'momentum' in ll:
                final_s.append(('momentum', ml_portfolios['momentum']))

        # 计算未来表现
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

    return summary, results


def find_optimal_parameters(returns_data, test_window=120, forward_window=10):
    """
    新增参数优化函数
    """
    alphas = [1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
    qs = [2.0, 2.5, 3.0, 3.5, 4.0]
    best_params = {'alpha': None, 'q': None}
    best_sharpe = float('-inf')

    for alpha in alphas:
        for q in qs:
            # 使用当前参数进行回测
            summary, _ = compare_optimization_methods(
                returns_data=returns_data,
                test_window=test_window,
                forward_window=forward_window,
                alpha=alpha,
                q=q,
                ll=['denoised']
            )

            if summary['denoised']['total_sharpe'] > best_sharpe:
                best_sharpe = summary['denoised']['total_sharpe']
                best_params = {'alpha': alpha, 'q': q}

    return best_params


def drop_consecutive_zeros(df, max_zeros=5):
    """
    删除包含超过指定数量连续0的列

    参数:
    - df: pandas DataFrame
    - max_zeros: 允许的最大连续0的数量

    返回:
    - 处理后的DataFrame
    """
    # 创建一个布尔掩码来标识连续0
    mask = df.eq(0).rolling(window=max_zeros + 1).sum().eq(max_zeros + 1)
    # 找出需要删除的列
    cols_to_drop = mask.any()
    # 返回清理后的数据
    return df.loc[:, ~cols_to_drop]


# 使用示例
if __name__ == '__main__':
    returns_data = pd.read_pickle('returns_matrix.pkl')

    # 使用不同的优化方法
    # methods = ['mean_variance', 'min_variance', 'risk_parity']
    # results = {}

    sharpe_result = {}
    return_result = {}

    for method in ['denoised', 'original', 'equal_weight', 'pca', 'cluster', 'sparse', 'momentum']:
        sharpe_result[method] = []
        return_result[method] = []

    # 进行20次随机测试
    for i in range(20):
        # 随机选取测试组合
        selected_portfolio = random.sample(list(returns_data.columns), 40)

        test_data = returns_data[selected_portfolio].fillna(0)

        # 取出有超过连续5个0数据的列
        test_data = drop_consecutive_zeros(test_data)

        # # 首先进行参数优化
        # print("开始参数优化...")
        # best_params = find_optimal_parameters(test_data)
        # print(f"最优参数: alpha={best_params['alpha']}, q={best_params['q']}")
        #
        # # 使用优化后的参数进行正式回测
        # summary, detailed_results = compare_optimization_methods(
        #     returns_data=test_data,
        #     test_window=120,
        #     forward_window=10,
        #     alpha=best_params['alpha'],
        #     q=best_params['q']
        # )

        print('第%d次测试' % i)
        print('组合：')
        print(selected_portfolio)
        summary, detailed_results = compare_optimization_methods(
            returns_data=test_data,
            test_window=240,
            forward_window=10,
            q=6,
            alpha=1.2,
            ll=['denoised', 'original', 'equal_weight', 'pca', 'cluster', 'sparse', 'momentum']
        )

        for method in ['denoised', 'original', 'equal_weight', 'pca', 'cluster', 'sparse', 'momentum']:
            sharpe_result[method].append(summary[method]['total_sharpe'])
            return_result[method].append(summary[method]['cum_return'])

        print("\n=== 实验结果对比 ===")
        for method in summary.keys():
            print(f"\n{method}策略:")
            for metric, value in summary[method].items():
                print(f"{metric}: {value:.4f}")

        # 进行统计检验
        # methods = list(summary.keys())
        # for i in range(len(methods)):
        #     for j in range(i + 1, len(methods)):
        #         method1, method2 = methods[i], methods[j]
        #         t_stat, p_value = stats.ttest_ind(
        #             detailed_results[method1]['sharpe'],
        #             detailed_results[method2]['sharpe']
        #         )
        #         print(f"\n{method1} vs {method2}夏普比率差异显著性检验:")
        #         print(f"t统计量: {t_stat:.4f}")
        #         print(f"p值: {p_value:.4f}")

        print('=========\n')

    total_return_d = pd.DataFrame(return_result)
    total_sharpe_d = pd.DataFrame(sharpe_result)
