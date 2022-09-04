import numpy as np
from scipy.stats import norm
from EM_algorithm import EM
import matplotlib.pyplot as plt

# Generate data
n_dim, n_clusters_data = 1, 2
n_train, n_test = 50, 50
n_data = n_train + n_test

np.random.seed(3562)
alpha_true = np.random.uniform(-3, 3, n_clusters_data)
np.random.seed(7965)
beta_true = np.random.uniform(-3, 3, n_clusters_data)

np.random.seed(123)
data_x = np.linspace(-3, 3, n_data)
np.random.seed(456)
data_sigma = np.random.uniform(low=0, high=1, size=n_clusters_data)

np.random.seed(789)
cluster_idx = np.random.choice([i for i in range(n_clusters_data)], size=n_data, replace=True)
data_weights = np.array([np.sum(cluster_idx == i) / cluster_idx.shape[0] for i in range(n_clusters_data)])
np.random.seed(657)
row_idx = np.random.choice([i for i in range(n_data)], size=n_data, replace=True)

np.random.seed(7568)
y = alpha_true + np.multiply(np.vstack((data_x for _ in range(n_clusters_data))).T, beta_true) +\
    np.random.randn(n_data, n_clusters_data) * data_sigma
data_x = data_x[row_idx]
data_y = y[row_idx, cluster_idx]

idx_shuffle = np.array([i for i in range(n_data)])
np.random.seed(964)
np.random.shuffle(idx_shuffle)
cluster_idx = cluster_idx[idx_shuffle]
data_x = data_x[idx_shuffle]
data_x = np.c_[np.ones(len(data_x)), data_x]
data_y = data_y[idx_shuffle]
X_train = data_x[:n_train]
y_train = data_y[:n_train]
X_test = data_x[n_train:]
y_test = data_y[n_train:]

# Initialize mixture
n_clusters = n_clusters_data  # To be optimized by wrapper of EM
np.random.seed(842)
mix_weights_init = np.random.dirichlet(np.ones(n_clusters), size=1).flatten()
np.random.seed(654)
mix_alpha_init = np.random.uniform(-3, 3, n_clusters)
np.random.seed(6324896)
mix_beta_init = np.random.uniform(-3, 3, n_clusters)
mix_coeff_init = np.c_[mix_alpha_init, mix_beta_init]
np.random.seed(159)
mix_sigma_init = np.random.uniform(0, 1, n_clusters)
mix_par_init = (mix_coeff_init, mix_sigma_init)

def log_like_fn(x, par):
    y_hat = x[:, :-1].dot(par[0])
    log_like = norm.logpdf(x[:, -1], loc=y_hat, scale=par[1])
    return log_like

data = np.c_[X_train, y_train]
em = EM(mix_weights_init, mix_par_init, log_like_fn, model_type='linear')
em.optimize(data, tol=1e-6)

# Visualize result
fig = plt.figure()
colors = ['blue', 'red', 'orange', 'purple', 'magenta']
x_grid = np.linspace(-5, 5, 10)
x_grid = np.c_[np.ones_like(x_grid), x_grid]
for i_cluster in [i for i in range(n_clusters)]:
    y_hat = np.dot(x_grid, em.mix_par[0][i_cluster])
    y_lower = y_hat - 1.64 * em.mix_par[1][i_cluster]
    y_upper = y_hat + 1.64 * em.mix_par[1][i_cluster]
    plt.scatter(X_train[cluster_idx[:n_train] == i_cluster, -1], y_train[cluster_idx[:n_train] == i_cluster], color=colors[i_cluster])
    plt.plot(x_grid[:, -1], y_hat, color='k')
    plt.fill_between(x_grid[:, -1], y_lower, y_upper, alpha=0.3, color='green')
plt.xlabel('Independent variable', fontsize=14)
plt.ylabel('Dependent variable', fontsize=14)
beta_OLS = np.dot(np.dot(np.linalg.inv(np.dot(X_train.T, X_train)), X_train.T), y_train)
y_hat_OLS = np.dot(x_grid, beta_OLS)
plt.plot(x_grid[:, -1], y_hat_OLS, linewidth=4, color='k', linestyle='--', label='OLS solution')
plt.legend(fontsize=12)
