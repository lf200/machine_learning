import numpy as np


class GaussianMixtureClusterEM:
    def gaussian(self, X, mu, cov):
        n = X.shape[1]  # dimension
        diff = (X - mu).T  # difference between X and mean vector mu
        gaussian_prob = np.diagonal(1 / ((2 * np.pi) ** (n / 2) * \
                                         np.linalg.det(cov) ** 0.5) \
                                    * np.exp(-0.5 * \
                                             np.dot(np.dot(diff.T, \
                                                           np.linalg.inv(cov)), \
                                                    diff))).reshape(-1, 1)
        return gaussian_prob

    def init_clusters(self, X, n_clusters):
        from sklearn.cluster import KMeans
        clusters = []
        kmeans = KMeans().fit(X)
        mu_k = kmeans.cluster_centers_
        for i in range(n_clusters):
            clusters.append({
                'alpha_k': 1.0 / n_clusters,
                'mu_k': mu_k[i],
                'cov_k': np.identity(X.shape[1], dtype=np.float64)
            })
        return clusters

    def expectation_step(self, X, clusters):
        totals = np.zeros((X.shape[0], 1), dtype=np.float64)
        for cluster in clusters:
            alpha_k = cluster['alpha_k']
            mu_k = cluster['mu_k']
            cov_k = cluster['cov_k']
            gamma_nk = (alpha_k * self.gaussian(X, mu_k, cov_k)).astype(np.float64)
            for i in range(X.shape[0]):
                totals[i] += gamma_nk[i]  # calculate denominator of Eq.(6.18)
            cluster['gamma_nk'] = gamma_nk
            cluster['totals'] = totals
        for cluster in clusters:
            cluster['gamma_nk'] /= cluster['totals']  # calcuate gamma,see Eq.(6.18)

    def maximization_step(self, X, clusters):
        N = float(X.shape[0])
        for cluster in clusters:
            gamma_nk = cluster['gamma_nk']
            cov_k = np.zeros((X.shape[1], X.shape[1]))
            N_k = np.sum(gamma_nk, axis=0)
            alpha_k = N_k / N  # see Eq.(6.22)
            mu_k = np.sum(gamma_nk * X, axis=0) / N_k  # see Eq.(6.19)
            for j in range(X.shape[0]):
                diff = (X[j] - mu_k).reshape(-1, 1)
                cov_k += gamma_nk[j] * np.dot(diff, diff.T)
            cov_k /= N_k  # see Eq.(6.20)
            cluster['alpha_k'] = alpha_k
            cluster['mu_k'] = mu_k
            cluster['cov_k'] = cov_k

    def get_likelihood(self, X, clusters):
        sample_likelihoods = np.log(np.array([cluster['totals'] for cluster in clusters]))
        return np.sum(sample_likelihoods), sample_likelihoods

    def fit(self, X, n_clusters, n_epochs):
        clusters = self.init_clusters(X, n_clusters)
        likelihoods = np.zeros((n_epochs,))
        scores = np.zeros((X.shape[0], n_clusters))
        history = []
        for i in range(n_epochs):
            clusters_snapshot = []
            for cluster in clusters:
                clusters_snapshot.append({'mu_k': cluster['mu_k'].copy(), 'cov_k': cluster['cov_k'].copy()})
            history.append(clusters_snapshot)
            self.expectation_step(X, clusters)
            self.maximization_step(X, clusters)
            likelihood, sample_likelihoods = self.get_likelihood(X, clusters)
            likelihoods[i] = likelihood
            print('Epoch: ', i + 1, 'Likelihood: ', likelihood)
        for i, cluster in enumerate(clusters):
            scores[:, i] = np.log(cluster['gamma_nk']).reshape(-1)
        return clusters, likelihoods, scores, sample_likelihoods, history


