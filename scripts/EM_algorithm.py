import numpy as np
from scipy.special import logsumexp
from types import FunctionType

class EM_wrapper:

    def EM_init_Kmeans(self):
        self.mix_par, self.weights = self.init_fn(data=self.X_train, n_clusters=self.n_clusters)

    def EM_init_random(self, random_seed=0):
        np.random.seed(random_seed)
        Z = np.random.choice([i for i in range(self.n_clusters)], size=self.n_train, replace=True)
        self.mix_par, self.weights = self.init_fn(data=self.X_train, Z=Z, n_clusters=self.n_clusters)

    def EM_init(self, init_method='Kmeans', random_seed=0):
        if init_method == 'Kmeans':
            self.EM_init_Kmeans()
        elif init_method == 'random':
            self.EM_init_random()
        else:
            raise Exception('No initlization method given')

class EM(EM_wrapper):

    def __init__(self, init_fn=None, mix_par_init=None, mix_weights_init=None, model_type='normal'):
        self.model_type = str.lower(model_type)
        # self.log_like_fn = log_like_fn
        self.bic = None
        self.aic = None
        if isinstance(init_fn, FunctionType):
            self.init_fn = init_fn
        elif not (mix_par_init is None or mix_weights_init is None):
            self.weights = mix_weights_init
            self.mix_par = mix_par_init
            self.init_fn = None
        else:
            raise Exception('No initilization given')

    def e_step(self):
        log_like_points = np.zeros((self.n_train, self.n_clusters))
        for i_cluster, par in enumerate(zip(*self.mix_par)):
            weight = self.weights[i_cluster]
            log_like_points[:, i_cluster] = np.log(weight) + self.log_like_fn(par) + np.log(1e-16)
        post = np.exp(log_like_points - logsumexp(log_like_points, axis=1)[:, None])
        return post, log_like_points.sum()

    def m_step(self, post, var_lower_bnd=1e-5):
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
                beta[i_cluster] = np.dot(np.dot(np.dot(np.linalg.inv(np.dot(np.dot(x_Z.T, w), x_Z)), x_Z.T), w), y_Z)
                resid = (y_Z - np.dot(x_Z, beta[i_cluster])) ** 2
                var[i_cluster] = np.dot(post_cluster[Z == i_cluster], resid)

            var /= post.sum(axis=0)

            var[np.where(np.isnan(var))] = var_lower_bnd
            var = np.maximum(var_lower_bnd, var)
            sigma = np.sqrt(var)

            mix_par = (beta, sigma)

        self.mix_par = mix_par
        self.weights = post.sum(axis=0) / self.n_train

    def train(self, data, n_clusters, init_method='deterministic', tol=1e-6, n_loops=int(1e+5), random_seed=0):

        self.convergence = False

        self.X_train = data
        self.n_train = self.X_train.shape[0]

        self.n_clusters = n_clusters

        if isinstance(self.init_fn, FunctionType):
            self.EM_init(init_method=init_method, random_seed=random_seed)

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
                self.post_train, self.log_like = self.e_step()
                break
        self.Z = np.argmax(self.post_train, axis=1)
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

