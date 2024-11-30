import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging

# データ準備
# 観測座標と値
x = np.array([0.1, 0.5, 0.8, 1.2, 1.5])
y = np.array([0.1, 0.5, 0.7, 1.0, 1.3])
z = np.array([1.0, 2.0, 1.5, 3.0, 2.5])

# セミバリオグラムとモデルに基づくクリギング補間
gridx = np.linspace(0, 2, 100)
gridy = np.linspace(0, 2, 100)

# Ordinary Krigingの適用
OK = OrdinaryKriging(
    x, y, z, variogram_model="exponential",
    verbose=False, enable_plotting=False
)
z_pred, ss = OK.execute("grid", gridx, gridy)

# 結果の可視化
plt.figure(figsize=(8, 6))
plt.contourf(gridx, gridy, z_pred, levels=50, cmap="viridis")
plt.colorbar(label="Interpolated value")
plt.scatter(x, y, c=z, edgecolor="k", s=100, label="Observations")
plt.title("Kriging Interpolation")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
