import numpy as np
import EM_edX as em
import kmeans_edX

X = np.loadtxt('..\\data\\Netflix edX project 4\\netflix_incomplete.txt')
X_gold = np.loadtxt('..\\data\\Netflix edX project 4\\netflix_complete.txt')
seeds = [0, 1, 2, 3, 4]
for K in [12]:
    log_like_lst = []
    for seed in seeds:
        mix, post = em.init(X, K, seed)
        mix, post, cost = kmeans_edX.run(X, mix, post)
        mixture, post, log_like = em.run(X, mix, post)
        log_like_lst.append(log_like)
        print(seed)
    log_like_max = max(log_like_lst)
    seed_min = seeds[np.argmax(np.array(log_like_lst))]
    print(log_like_max)
    mix, post = em.init(X, K, seed_min)
    mix, post, cost = kmeans_edX.run(X, mix, post)
    mixture, post, log_like = em.run(X, mix, post)

X_hat = em.fill_matrix(X, mixture)

rmse = np.linalg.norm(X_hat - X_gold)