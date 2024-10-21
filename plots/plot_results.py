'''
author: yihang
date: 2024/10/15
function: visualize the results
'''


import numpy as np
import matplotlib.pyplot as plt
import os


models = ['dragonnet', 'nednet', 'tarnet']
base_path = "D:/ZhangYihang/Dragonnet/dragonnet/result/ihdp/"
model = 'dragonnet'
sample = '10'
result_path = f"D:/ZhangYihang/Dragonnet/{model}/result/ihdp/dragonnet"
test_result = np.load(os.path.join(result_path, f"{sample}/targeted_regularization/0_replication_test.npz"))
q_t0 = test_result['q_t0']
q_t1 = test_result['q_t1']
g = test_result['g']
t = test_result['t'].flatten()
y = test_result['y'].flatten()

simulation_data = np.load(os.path.join(result_path, f"{sample}/simulation_outputs.npz"))
mu_0 = simulation_data['mu_0']
mu_1 = simulation_data['mu_1']
test_index = test_result['index']
true_ate = (mu_1[test_index.flatten()] - mu_0[test_index.flatten()]).mean()

# 1. Plot potential outcomes under control and treatment
plt.figure(figsize=(10, 5))
plt.scatter(range(len(q_t0)), q_t0, alpha=0.5, label='Potential Outcome under Control (q_t0)', color='blue')
plt.scatter(range(len(q_t1)), q_t1, alpha=0.5, label='Potential Outcome under Treatment (q_t1)', color='red')
plt.xlabel('Sample Index')
plt.ylabel('Potential Outcomes')
plt.legend()
plt.title('Potential Outcomes under Control and Treatment')
plt.show()

# 2. Plot Individual Treatment Effects (ITE)
ite = q_t1 - q_t0
plt.figure(figsize=(12, 6))
plt.hist(ite, bins=30, color='purple', alpha=0.7)
plt.xlabel('ITE (q_t1 - q_t0)')
plt.ylabel('Frequency')
plt.title('Distribution of Individual Treatment Effects (ITE)')
plt.show()

# 3. Plot the treatment propensity scores
plt.figure(figsize=(12, 6))
plt.hist(g, bins=30, color='green', alpha=0.7)
plt.xlabel('Propensity Score (g)')
plt.ylabel('Frequency')
plt.title('Distribution of Treatment Propensity Scores')
plt.show()

# 4. Plot observed outcomes vs. predicted treatment outcomes
predicted_outcomes = (t * q_t1) + ((1 - t) * q_t0)
plt.figure(figsize=(12, 6))
plt.scatter(predicted_outcomes, y, alpha=0.5, color='orange')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Predicted Outcomes')
plt.ylabel('Observed Outcomes')
plt.title('Observed vs. Predicted Outcomes')
plt.show()

# 5. Plot ATE comparison between predicted and true values
pred_ate = ite.mean()
ate_values = [pred_ate, true_ate]
categories = ['Predicted ATE', 'True ATE']
colors = ['skyblue', 'salmon']

plt.figure(figsize=(8, 5))
bars = plt.bar(categories, ate_values, color=colors)

difference = abs(pred_ate - true_ate)
for i, bar in enumerate(bars):
    if bar.get_height() == max(ate_values):
        plt.gca().add_patch(plt.Rectangle(
            (bar.get_x(), min(ate_values)),
            bar.get_width(),
            difference,
            color='lightcoral',
            alpha=0.5
        ))

plt.ylabel('ATE Value')
plt.title(f'[{model}] Comparison of Predicted and True ATE with Difference Highlighted')
plt.show()

# 6. Additional scatter plot of ITE vs True ATE
plt.figure(figsize=(10, 6))
plt.scatter(range(len(ite)), ite, alpha=0.5, label='Predicted ITE', color='purple')
plt.axhline(y=true_ate, color='orange', linestyle='--', label='True ATE')
plt.xlabel('Sample Index')
plt.ylabel('Individual Treatment Effect (ITE)')
plt.legend()
plt.title(f'[{model}] Individual Treatment Effects (ITE) with True ATE Line')
plt.show()