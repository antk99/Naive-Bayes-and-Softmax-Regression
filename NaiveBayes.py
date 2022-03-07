import threading

import numpy as np

# TODO verify/test algorithm implementation correctness [remaining: logsumexp, predict - posterior calculations]
# TODO finish commenting & documenting code
# TODO Experimentation on datasets [remaining: sentiment dataset]

from sklearn import model_selection
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer


def logsumexp(Z):  # dimension C x N
    Zmax = np.max(Z, axis=0)[None, :]  # max over C
    log_sum_exp = Zmax + np.log(np.sum(np.exp(Z - Zmax), axis=0))
    return log_sum_exp


class NaiveBayes:
    def __init__(self):
        pass

    def fit(self, x, y):
        N, D = x.shape
        self.C = np.max(y) + 1
        # one parameter for each feature conditioned on each class
        probs = np.zeros((self.C, D))
        Nc = np.zeros(self.C)  # number of instances in class c

        # for each class get the MLE for each d,c (rel frequencies)
        for c in range(self.C):
            x_c = x[y == c]  # slice all the elements from class c
            Nc[c] = x_c.shape[0]  # get number of elements of class c

            num_threads = 10

            interval_size = int(D / num_threads)
            threads = []
            for thread in range(num_threads):
                threads.append(
                    threading.Thread(
                        target=NaiveBayes._fit_thread,
                        args=(x_c, c, thread*interval_size, (thread*interval_size) + interval_size, probs))
                )
                threads[thread].start()

            for thread in threads:
                thread.join()

            # without threading
            #for d in range(D):
                # count_d = np.sum(x_c[:, d]) + 1     # counts of word d in all documents labelled c
                # tot_count = np.sum(x_c)             # total word count in all documents labelled c
                # probs[c][d] = count_d / tot_count   # MLE for each d,c (rel frequency)

        self.probs = probs  # C x D
        self.pi = (Nc + 1) / (N + self.C)  # Laplace smoothing (using alpha_c=1 for all c)
        return self

    def _fit_thread(x_c, c, d_start, d_end, probs):
        for d in range(d_start, d_end):
            count_d = np.sum(x_c[:, d]) + 1  # counts of word d in all documents labelled c
            tot_count = np.sum(x_c)  # total word count in all documents labelled c
            probs[c][d] = count_d / tot_count  # MLE for each d,c (rel frequency)

    def predict(self, xt):
        Nt, D = xt.shape
        # for numerical stability we work in the log domain
        # we add a dimension because this is added to the log-likelihood matrix
        # that assigns a likelihood for each class (C) to each test point, and so it is C x N
        log_prior = np.log(self.pi)[:, None]

        log_likelihood = np.zeros((Nt, self.C))
        for i in range(Nt):
            a = np.log(self.probs ** xt[i])
            b = np.sum(a, axis=1)
            log_likelihood[i] = b

        log_posterior = log_prior.T + log_likelihood
        posterior = np.exp(log_posterior - logsumexp(log_posterior))
        return posterior  # dimension N x C


if __name__ == '__main__':
    twenty_train = fetch_20newsgroups(subset='train',
                                      shuffle=True,
                                      remove=(['headers', 'footers', 'quotes']))

    with open("stopwords.txt") as f:
        stop_words = f.read()
        stop_words = stop_words.split('\n')

        vectorizer = CountVectorizer(max_features=5000, stop_words=stop_words)
    X_counts = vectorizer.fit_transform(twenty_train.data)

    from sklearn.feature_extraction.text import TfidfTransformer
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_counts)
    X_tf = tf_transformer.transform(X_counts)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_counts)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(X_train_tfidf, twenty_train.target, test_size=0.2)

    model = NaiveBayes()
    model.fit(x_train.toarray(), y_train)
    y_prob = model.predict(x_test.toarray())
    y_pred = np.argmax(y_prob, axis=1)
    accuracy = np.sum(y_pred == y_test) / y_pred.shape[0]
    print(f"Accuracy is {accuracy}")

