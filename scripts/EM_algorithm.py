import numpy as np
from scipy.special import logsumexp

class EM:
    def __init__(self, mix_weights_init, mix_par_init, log_like_fn, model_type='normal'):
        self.weights = mix_weights_init
        self.mix_par = mix_par_init
        self.model_type = str.lower(model_type)
        self.n_clusters = self.weights.shape[0]
        self.log_like_fn = log_like_fn

    def e_step(self):
        n_train = self.data_train.shape[0]
        log_like_points = np.empty((n_train, self.n_clusters))
        for i_cluster, par in enumerate(zip(*self.mix_par)):
            weight = self.weights[i_cluster]
            log_like_points[:, i_cluster] = self.log_like_fn(self.data_train, par) + np.log(weight)
        post = log_like_points - logsumexp(log_like_points, axis=1)[:, None]
        return np.exp(post), log_like_points.sum()

    def m_step(self, post):
        n_train = self.data_train.shape[0]
        mix_par = ()
        if self.model_type == 'normal':
            mu = np.dot(post.T, self.data_train) / post.sum(axis=0)[:, None]
            var = np.zeros_like(mu)
            for i_cluster in range(self.n_clusters):
                var[i_cluster] = np.dot(post[:, i_cluster].T, (self.data_train - mu[i_cluster]) ** 2)
            # var = np.multiply(post, (np.tile(self.data_train, (1, self.n_clusters)) - mu.T)**2).sum(axis=0)
            var /= post.sum(axis=0)[:, None]
            sigma = np.sqrt(var)
            mix_par = (mu, sigma)
        elif self.model_type == 'multivariate_normal':
            pass
        elif self.model_type == 'poisson':
            pass
        elif self.model_type == 'linear':
            pass
        self.mix_par = mix_par
        self.weights = post.sum(axis=0) / n_train

    def optimize(self, data_train, tol=1e-6, n_loops=int(1e+5)):
        self.convergence = False
        self.data_train = data_train

        old_log_like = None
        new_log_like = None
        post = None
        for _ in range(n_loops):
            if (old_log_like is None) or (np.abs(new_log_like-old_log_like) >= tol * np.abs(new_log_like)):
                old_log_like = new_log_like
                post, new_log_like = self.e_step()
                self.m_step(post)
            else:
                self.convergence = True
                self.log_like = new_log_like
                self.post_train = post
                break
        self.calc_bic()

    def calc_bic(self):
        self.bic = None

    def predict(self, data_test):
        self.data_test = data_test

if __name__ == '__main__':

    from scipy.stats import norm
    import matplotlib.pyplot as plt

    # Generate data
    n_dim, n_clusters_data = 1, 2
    n_train, n_test = 100, 100
    n_data = n_train + n_test

    np.random.seed(123)
    data_mu = np.random.uniform(low=-1, high=1, size=(n_clusters_data, n_dim))
    np.random.seed(456)
    data_sigma = np.random.uniform(low=0, high=1, size=(n_clusters_data, n_dim))
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
    data_train = data[:n_train]
    data_test = data[n_train:]

    # Initialize mixture
    n_clusters = n_clusters_data  # To be optimized by wrapper of EM
    np.random.seed(321)
    mix_weights_init = np.random.dirichlet(np.ones(n_clusters), size=1).flatten()
    np.random.seed(654)
    mix_mu_init = np.random.uniform(low=-10, high=10, size=n_clusters)
    np.random.seed(159)
    mix_sigma_init = np.random.uniform(low=0, high=10, size=n_clusters)
    mix_par_init = (mix_mu_init, mix_sigma_init)

    # EM algorithm for a single initialization
    log_like_fn = lambda x, par: norm.logpdf(x, loc=par[0], scale=par[1]).squeeze()  # Set log-likelihood function based on mixture distribution
    em = EM(mix_weights_init, mix_par_init, log_like_fn)
    em.optimize(data_train, tol=1e-15)

    # Visualize result
    fig = plt.figure()
    bins = np.linspace(data.min(), data.max(), 20)
    # plt.hist(data, bins=bins, alpha=0.5, color='blue')
    plt.hist(data[cluster_idx == 0], bins=bins, alpha=0.3, color='blue')
    plt.hist(data[cluster_idx == 1], bins=bins, alpha=0.3, color='green')
    for (mu, mu_real) in zip(em.mix_par[0], data_mu):
        plt.axvline(mu, color='black')
        plt.axvline(mu_real, color='red')

