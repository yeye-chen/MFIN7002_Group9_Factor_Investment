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
        if lambda_val > lambda_plus:
            # 对大于上界的特征值进行更强的压缩
            return 1 + (lambda_val - 1) / (alpha * 2 * (1 + (lambda_val - 1)))
        elif lambda_val < lambda_minus:
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


def generate_weights(return_d, date, q=6, alpha=1.2, test_window=240, ll=None):
    if ll is None:
        ll = ['pca']

    hist_data = return_d[return_d.index <= date].tail(test_window)

    # derive denoise covariance matrix
    squeezed_corr = squeeze_correlation_matrix(hist_data, alpha=alpha, q=q)

    # calculate the weights
    ml_portfolios = build_ml_portfolios(hist_data, squeezed_corr, lookback=min(60, test_window))

    return ml_portfolios['pca']


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



if __name__ == '__main__':
    # import daily return data, at least contains 240 trading date
    return_data = pd.read_pickle('returns_matrix.pkl')

    # import next week's portfolio
    portfolio_s = list(pd.read_pickle()['Ticker'])  # make sure is a list

    # get last Friday's date
    time_point = pd.to_datetime('2024-12-20')
    # extract the return's of stocks in portfolio
    return_p = return_data[portfolio_s]
    return_p = return_p.fillna(0)

    return_p = drop_consecutive_zeros(return_p)

    # derive the weights
    weights = generate_weights(return_p, time_point)
    weights.to_csv('weights.csv')
    print(weights)
