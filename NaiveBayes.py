import numpy as np

# TODO verify/test algorithm implementation correctness
# TODO finish commenting & documenting code
# TODO Experimentation on datasets


def logsumexp(Z):  # dimension C x N
    Zmax = np.max(Z, axis=0)[None, :]  # max over C
    log_sum_exp = Zmax + np.log(np.sum(np.exp(Z - Zmax), axis=0))
    return log_sum_exp


class NaiveBayes:
    def __init__(self):
        pass

    def fit(self, x, y):
        N, D = x.shape
        C = np.max(y) + 1
        # one parameter for each feature conditioned on each class
        probs = np.zeros((C, D))
        Nc = np.zeros(C)  # number of instances in class c

        # for each class get the MLE for each d,c (rel frequencies)
        for c in range(C):
            x_c = x[y == c]  # slice all the elements from class c
            Nc[c] = x_c.shape[0]  # get number of elements of class c

            for d in range(D):
                count_d = np.sum(x_c[:, d]) + 1     # counts of word d in all documents labelled c
                tot_count = np.sum(x_c)             # total word count in all documents labelled c
                probs[c][d] = count_d / tot_count   # MLE for each d,c (rel frequency)

        self.probs = probs  # C x D
        self.pi = (Nc + 1) / (N + C)  # Laplace smoothing (using alpha_c=1 for all c)
        return self

    def predict(self, xt):
        Nt, D = xt.shape
        # for numerical stability we work in the log domain
        # we add a dimension because this is added to the log-likelihood matrix
        # that assigns a likelihood for each class (C) to each test point, and so it is C x N
        log_prior = np.log(self.pi)[:, None]

        log_likelihood = np.zeros((Nt, D))
        for i in range(Nt):
            log_likelihood[i] = np.prod((np.log(self.probs) ** xt[i]), axis=1)

        log_posterior = log_prior + log_likelihood
        posterior = np.exp(log_posterior - logsumexp(log_posterior))
        return posterior.T  # dimension N x C