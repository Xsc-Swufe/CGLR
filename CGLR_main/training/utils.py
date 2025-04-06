import numpy as np
import torch
import pickle
import pandas as pd

def load_data():
    #code_num = 772

    #code_num = 772

    code_num = 305
    fts = 9
    f = open(r'../data/CAS_B.csv')
    df = pd.read_csv(f, header=None)
    data = df.iloc[:, 0:-1].values
    eod_data = data.reshape(-1, code_num, fts)
    data_label = df.iloc[:, -1].values
    ground_truth = data_label.reshape(code_num, -1)

    return eod_data, ground_truth

def R2_score_calculate(true, pred):
    r2 = 1 - np.sum((true - pred) ** 2) / np.sum(true ** 2)
    return r2

def IC_ICIR_score_calculate(true, pred, Length_time):
    df = pd.DataFrame()
    times = [time for time in list(range(1, Length_time + 1)) for i in range(int(len(true) / Length_time))]
    df['date'] = times
    df['true'] = true
    df['pred'] = pred
    rank_ic = df.groupby("date").apply(lambda x: x["true"].corr(x["pred"], method="spearman"))
    rank_ic = np.array(rank_ic)
    rank_ic_mean = np.mean(rank_ic)
    rank_ic_std = np.std(rank_ic, ddof=1)
    return rank_ic_mean, rank_ic_mean/rank_ic_std


def compute_lead_lag(data, window_size):
    """
    计算滞后步和平均滞后时间差。

    参数：
        data (np.ndarray or torch.Tensor):  (T, N, F)，
        window_size (int):

    返回：
        lag_steps (np.ndarray):  (N, N)。
        avg_lag_diff (np.ndarray):  (N, N)。
    """

    '''
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
        '''
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    T, N, F = data.shape


    lag_steps = np.zeros((N, N))
    avg_lag_diff = np.zeros((N, N))
    data_normalized = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    # 对每对个体 (i, j) 计算滞后步和平均滞后时间差
    for i in range(N):
        for j in range(N):
            if i != j:
               
                x = data_normalized[:, i, :].mean(axis=-1)
                y = data_normalized[:, j, :].mean(axis=-1)

                
                X_fft = np.fft.fft(x)
                Y_fft = np.fft.fft(y)

                # 计算交叉功率谱密度
                cross_spectrum = X_fft * np.conj(Y_fft)

                # 计算互相关
                cross_correlation = np.fft.ifft(cross_spectrum).real

                # 限制在窗口范围内
                mid = len(cross_correlation) // 2
                relevant_range = np.concatenate((cross_correlation[-window_size:], cross_correlation[:window_size + 1]))

                # 计算滞后步 (最优滞后步)
                optimal_lag = np.argmax(np.abs(relevant_range)) - window_size

                # 计算平均滞后时间差
                abs_corr = np.abs(relevant_range)
                avg_lag = np.sum(np.arange(-window_size, window_size + 1) * abs_corr) / np.sum(abs_corr)

                lag_steps[i, j] = optimal_lag
                avg_lag_diff[i, j] = avg_lag

    return lag_steps, avg_lag_diff
