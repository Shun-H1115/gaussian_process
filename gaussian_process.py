import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# データの準備
def f(x):
    return np.sin(x)


if __name__ == "__main__":
    # 訓練データ
    X_train = np.array([[1], [3], [5], [6], [8]])
    y_train = f(X_train).ravel()

    # テストデータ
    X_test = np.linspace(0, 10, 100).reshape(-1, 1)

    # カーネルの定義
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))

    # ガウス過程回帰モデルの作成
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)

    # モデルの学習
    gp.fit(X_train, y_train)

    # 予測
    y_pred, sigma = gp.predict(X_test, return_std=True)

    # 結果の可視化
    plt.figure()
    plt.plot(X_test, f(X_test), 'r:', label="True function")
    plt.errorbar(X_train, y_train, yerr=0.1, fmt='r.', markersize=10, label="Observations")
    plt.plot(X_test, y_pred, 'b-', label="Prediction")
    plt.fill_between(X_test.ravel(), 
                    y_pred - 1.96 * sigma, 
                    y_pred + 1.96 * sigma, 
                    alpha=0.5, label="95% confidence interval")
    plt.legend()
    plt.show()