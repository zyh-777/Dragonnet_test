import numpy as np
from sklearn.metrics import mean_squared_error

# 加载数据
result_path = "D:/ZhangYihang/Dragonnet/dragonnet/result/ihdp/dragonnet/0/targeted_regularization/0_replication_test.npz"
data = np.load(result_path)

q_t0 = data['q_t0']
q_t1 = data['q_t1']
y = data['y'].flatten()
t = data['t'].flatten()

ite = q_t1 - q_t0
ate = np.mean(ite)

print("Individual Treatment Effect (ITE) Mean:", np.mean(ite))
print("Average Treatment Effect (ATE):", ate)

predicted_outcomes = (t * q_t1) + ((1 - t) * q_t0)
mse = mean_squared_error(y, predicted_outcomes)
print("MSE between observed outcomes and predicted outcomes:", mse)
