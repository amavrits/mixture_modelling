import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans

class EM:
    def __init__(self, mix_weights_init, mix_par_init, log_like_fn, model_type='normal'):
        self.weights = mix_weights_init
        self.mix_par = mix_par_init
        self.model_type = str.lower(model_type)
        self.n_clusters = self.weights.shape[0]
        self.log_like_fn = log_like_fn
        self.bic = None
        self.aic = None

    def e_step(self):
        log_like_points = np.empty((self.n_train, self.n_clusters))
        for i_cluster, par in enumerate(zip(*self.mix_par)):
            weight = self.weights[i_cluster]
            log_like_points[:, i_cluster] = np.log(weight) + self.log_like_fn(self.X_train, par)
        post = np.exp(log_like_points - logsumexp(log_like_points, axis=1)[:, None])
        return post, log_like_points.sum()

    def m_step(self, post):
        mix_par = ()
        if self.model_type == 'normal':
            mu = np.dot(post.T, self.X_train) / post.sum(axis=0)[:, None]
            var = np.zeros_like(mu)
            for i_cluster in range(self.n_clusters):
                var[i_cluster] = np.dot(post[:, i_cluster].T, (self.X_train - mu[i_cluster]) ** 2)
            # var = np.multiply(post, (np.tile(self.X_train, (1, self.n_clusters)) - mu.T)**2).sum(axis=0)
            var /= post.sum(axis=0)[:, None]
            sigma = np.sqrt(var)
            sigma = np.maximum(1e-6, sigma)
            mix_par = (mu, sigma)
        elif self.model_type == 'multivariate_normal':
            pass
        elif self.model_type == 'poisson':
            pass
        elif self.model_type == 'linear':
            pass
        self.mix_par = mix_par
        self.weights = post.sum(axis=0) / self.n_train

    def optimize(self, X_train, tol=1e-6, n_loops=int(1e+5)):
        self.convergence = False
        self.X_train = X_train
        self.n_train = self.X_train.shape[0]

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
        self.calc_aic()

    def calc_bic(self):
        d = self.n_clusters * len(self.mix_par) + self.weights.shape[0] - 1  # Number of model parameters
        bic = np.log(self.n_train) * d - 2 * self.log_like
        self.bic = bic  # The model with the lowest BIC wins

    def calc_aic(self):
        d = self.n_clusters * len(self.mix_par) + self.weights.shape[0] - 1  # Number of model parameters
        aic = 2 * d - 2 * self.log_like
        self.aic = aic  # The model with the lowest AIC wins

    def predict(self, X_test):
        self.X_test = X_test

if __name__ == '__main__':

    from scipy.stats import norm
    import matplotlib.pyplot as plt

    # Generate data
    n_dim, n_clusters_data = 1, 2
    n_train, n_test = 100, 100
    n_data = n_train + n_test

    np.random.seed(123)
    # data_mu = np.random.uniform(low=-1, high=1, size=(n_clusters_data, n_dim))
    data_mu = np.linspace(0, 1, n_clusters_data)[:, None]
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

    log_like_fn = lambda x, par: norm.logpdf(x, loc=par[0], scale=par[1]).squeeze()  # Set log-likelihood function based on mixture distribution

    # EM algorithm for a single initialization
    em = EM(mix_weights_init, mix_par_init, log_like_fn)
    em.optimize(X_train, tol=1e-15)

    # Visualize result
    fig = plt.figure()
    bins = np.linspace(data.min(), data.max(), 20)
    plt.hist(data, density=True, bins=bins, alpha=0.3, color='blue')
    # for i_cluster in range(n_clusters_data):
    #     plt.hist(data[cluster_idx == i_cluster], density=True, bins=bins, alpha=0.3)
    for (mu, mu_real) in zip(em.mix_par[0], data_mu):
        plt.axvline(mu, color='black')
        plt.axvline(mu_real, color='red')


    #TODO: Implement test to validate script with scikit
    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(n_components=n_clusters, random_state=0).fit(X_train)
    gm.bic(X_train)
    gm.aic(X_train)
    sk_log_like = gm.score_samples(X_train).sum()
    sk_mus = gm.means_
    sk_sigmas = np.sqrt(gm.covariances_).squeeze()

    mu_comparison = np.c_[np.sort(sk_mus.squeeze()), np.sort(em.mix_par[0].squeeze())]
