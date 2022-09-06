import numpy as np
from scipy.stats import norm
from EM_algorithm import EM
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class EM_GMM(EM):

    def log_like_fn(self, par):
        x = self.X_train
        return norm.logpdf(x, loc=par[0], scale=par[1]).squeeze()


# Generate data
n_dim, n_clusters_data = 1, 2
n_train, n_test = 100, 100
n_data = n_train + n_test

np.random.seed(123)
data_mu = np.linspace(0, 10, n_clusters_data)[:, None]
np.random.seed(456)
data_sigma = np.random.uniform(low=0, high=3, size=(n_clusters_data, n_dim))
np.random.seed(789)
cluster_idx = np.random.choice([i for i in range(n_clusters_data)], size=n_data, replace=True)
data_weights = np.array([np.sum(cluster_idx == i) / cluster_idx.shape[0] for i in range(n_clusters_data)])
np.random.seed(9827)
data = data_mu[cluster_idx] + data_sigma[cluster_idx] * np.random.randn(n_data, n_dim)
idx_shuffle = np.array([i for i in range(len(data))])
np.random.seed(964)
np.random.shuffle(idx_shuffle)
cluster_idx = cluster_idx[idx_shuffle]
data = data[idx_shuffle]
X_train = data[:n_train]
X_test = data[n_train:]

# Initialize mixture
n_clusters = n_clusters_data  # To be optimized by wrapper of EM
np.random.seed(321)
mix_weights_init = np.random.dirichlet(np.ones(n_clusters), size=1).flatten()
np.random.seed(654)
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_train)
mix_mu_init = kmeans.cluster_centers_
np.random.seed(159)
mix_sigma_init = np.zeros(n_clusters)
for i_cluster in range(n_clusters):
    mix_sigma_init[i_cluster] = np.std(X_train[kmeans.labels_ == i_cluster])
mix_sigma_init = np.maximum(1e-6, mix_sigma_init)
mix_par_init = (mix_mu_init, mix_sigma_init)

# EM algorithm for a single initialization
em = EM_GMM(mix_weights_init=mix_weights_init, mix_par_init=mix_par_init)
em.train(X_train, n_clusters=2, tol=1e-15)

# Visualize result
fig = plt.figure()
bins = np.linspace(data.min(), data.max(), 20)
plt.hist(data, density=True, bins=bins, alpha=0.3, color='blue')
for (mu, mu_real) in zip(em.mix_par[0], data_mu):
    plt.axvline(mu, color='black')
    plt.axvline(mu_real, color='red')


