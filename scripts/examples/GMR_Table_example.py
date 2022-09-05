import numpy as np
import pandas as pd
from scipy.stats import norm
from EM_algorithm import EM
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


def init_fn(data, n_clusters, Z=None):
    x = data[:, 1].reshape(-1, 1)
    y = data[:, -1]
    if Z is None:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(np.c_[x, y])
        Z = kmeans.labels_
    betas = np.zeros((n_clusters, x.shape[1]+1))
    sigma = np.zeros(n_clusters)
    for i_cluster in range(n_clusters):
        idx = np.where(Z == i_cluster)[0]
        reg = LinearRegression().fit(x[idx], y[idx])
        y_hat = reg.predict(x[idx])
        residuals = (y[idx] - y_hat) ** 2
        sigma[i_cluster] = np.sqrt(np.dot(residuals.T, residuals) / (x.shape[0] - x.shape[1]))
        betas[i_cluster] = np.array([reg.intercept_, reg.coef_[0]])
    mix_par_init = (betas, sigma)
    mix_weights_init = np.array([sum(Z == i_cluster) for i_cluster in range(n_clusters)]) / x.shape[0]
    return mix_par_init, mix_weights_init

def log_like_fn(x, par):
    y_hat = x[:, :-1].dot(par[0])
    log_like = norm.logpdf(x[:, -1], loc=y_hat, scale=par[1])
    return log_like

def logistic_prediction(x, theta):
    x = np.c_[np.ones(x.shape[0]), x]
    y = 1 / (1 + np.exp(-np.dot(x, theta)))
    Z = np.where(y <= 0.5, 0, 1)
    return Z


# Generate data
n_dim, n_clusters_data = 1, 2
n_train, n_test = 20, 50
n_data = n_train + n_test

np.random.seed(3654123)
mu_true = np.random.uniform(-5, 5, (n_clusters_data, n_dim))
np.random.seed(9856321)
sigma_true = np.random.uniform(0, 2, (n_clusters_data, n_dim))
table = np.zeros((n_data, n_dim))
for i_cluster in range(n_clusters_data):
    np.random.seed(32123+i_cluster*1_000)
    idx = np.arange(i_cluster * int(n_data/n_clusters_data), (i_cluster+1)*int(n_data/n_clusters_data))
    table[idx] = mu_true[i_cluster] + sigma_true[i_cluster] * np.random.randn(int(n_data/n_clusters_data), n_dim)
np.random.seed(32123)
theta_true = np.random.uniform(-5, 5, n_dim+1)
cluster_idx = logistic_prediction(table, theta_true)

np.random.seed(3562)
alpha_true = np.random.uniform(-3, 3, n_clusters_data)
np.random.seed(7965)
beta_true = np.random.uniform(-3, 3, n_clusters_data)
np.random.seed(123)
data_x = np.linspace(-3, 3, n_data)
np.random.seed(456)
data_sigma = np.random.uniform(low=0, high=1, size=n_clusters_data)

np.random.seed(789)
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
table = table[idx_shuffle]
X_train = data_x[:n_train]
y_train = data_y[:n_train]
X_test = data_x[n_train:]
y_test = data_y[n_train:]
table_train = table[:n_train]
table_test = table[n_train:]

data = np.c_[X_train, y_train]
em = EM(log_like_fn, init_fn=init_fn, model_type='linear')
em.train(data, init_method='random', n_clusters=n_clusters_data, tol=1e-6)


clf = LogisticRegression(random_state=0).fit(table_train, em.Z)

df_thetas = pd.DataFrame(data=np.c_[theta_true.reshape(-1, 1), np.array([clf.intercept_.item(), clf.coef_.squeeze()])],
                         columns=['True', 'Estimated'], index=['intercept'] + ['slope_'+str(i) for i in range(clf.coef_.shape[0])])

well_classified_rate = np.sum(cluster_idx[:n_train] == em.Z) / n_train
mu = np.dot(em.post_train.T, table_train) / np.sum(em.post_train, axis=0)[:, None]


x = np.linspace(-3, 3, 1_000)
x = np.c_[np.ones(1_000), x]
y_model_true = 1 / (1 + np.exp(-np.dot(x, df_thetas['True'].values)))
y_model_estimated = 1 / (1 + np.exp(-np.dot(x, df_thetas['Estimated'].values)))

fig = plt.figure()
plt.plot(x[:, -1], y_model_true, color='b')
plt.scatter(X_train[:, -1], cluster_idx[:n_train], color='b')
plt.plot(x[:, -1], y_model_estimated, color='r')
plt.scatter(X_train[:, -1], em.Z, color='r')


## Visualize result
# fig = plt.figure()
# colors = ['blue', 'red', 'orange', 'purple', 'magenta']
# x_grid = np.linspace(-5, 5, 10)
# x_grid = np.c_[np.ones_like(x_grid), x_grid]
# for i_cluster in [i for i in range(em.n_clusters)]:
#     y_hat = np.dot(x_grid, em.mix_par[0][i_cluster])
#     y_lower = y_hat - 1.64 * em.mix_par[1][i_cluster]
#     y_upper = y_hat + 1.64 * em.mix_par[1][i_cluster]
#     plt.scatter(X_train[cluster_idx[:n_train] == i_cluster, -1], y_train[cluster_idx[:n_train] == i_cluster], color=colors[i_cluster])
#     plt.plot(x_grid[:, -1], y_hat, color='k')
#     plt.fill_between(x_grid[:, -1], y_lower, y_upper, alpha=0.3, color='green')
# plt.xlabel('Independent variable', fontsize=14)
# plt.ylabel('Dependent variable', fontsize=14)
# beta_OLS = np.dot(np.dot(np.linalg.inv(np.dot(X_train.T, X_train)), X_train.T), y_train)
# y_hat_OLS = np.dot(x_grid, beta_OLS)
# plt.plot(x_grid[:, -1], y_hat_OLS, linewidth=4, color='k', linestyle='--', label='OLS solution')
# plt.legend(fontsize=12)

