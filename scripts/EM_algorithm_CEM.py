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
        log_like_points = np.zeros((self.n_train, self.n_clusters))
        for i_cluster, par in enumerate(zip(*self.mix_par)):
            weight = self.weights[i_cluster]
            log_like_points[:, i_cluster] = np.log(weight) + self.log_like_fn(self.X_train, par) + np.log(1e-16)
        post = np.exp(log_like_points - logsumexp(log_like_points, axis=1)[:, None])
        return post, log_like_points.sum()

    def m_step(self, post, var_lower_bnd=1e-3):
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

            x = self.X_train[:, :-1]
            y = self.X_train[:, -1]
            Z = np.argmax(post, axis=1)
            n_dim = x.shape[1] - 1 + 1 # -1 because the first column is all ones and +1 for variance
            beta = np.zeros((self.n_clusters, n_dim))
            var = np.zeros(self.n_clusters)

            for i_cluster in range(self.n_clusters):

                post_cluster = post[:, i_cluster]
                if np.all(post_cluster == 0):
                    continue

                ## EM: Simple EM
                # w = np.diag(post_cluster)
                # beta[:, i_cluster] = np.dot(np.dot(np.dot(np.linalg.inv(np.dot(np.dot(x.T, w), x)), x.T), w), y)
                # resid = (y - np.dot(x, beta[:, i_cluster])) ** 2
                # var[i_cluster] = np.dot(post_cluster, resid) / post_cluster.sum()
                # sigma = np.sqrt(var)

                # CEM: EM with classification step
                if sum(Z == i_cluster) == 0:
                    continue
                x_Z = x[Z == i_cluster]
                y_Z = y[Z == i_cluster]
                w = np.diag(post_cluster[Z == i_cluster])
                beta[i_cluster, :] = np.dot(np.dot(np.dot(np.linalg.inv(np.dot(np.dot(x_Z.T, w), x_Z)), x_Z.T), w), y_Z)
                resid = (y_Z - np.dot(x_Z, beta[i_cluster, :])) ** 2
                var[i_cluster] = np.dot(post_cluster[Z == i_cluster], resid)

            var /= post.sum(axis=0)

            var_lower_bnd = 1
            var[np.where(np.isnan(var))] = var_lower_bnd
            var = np.maximum(var_lower_bnd, var)
            sigma = np.sqrt(var)

            mix_par = (beta, sigma)

        self.mix_par = mix_par
        self.weights = post.sum(axis=0) / self.n_train

    def optimize(self, data, tol=1e-6, n_loops=int(1e+5)):

        self.convergence = False

        self.X_train = data
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
