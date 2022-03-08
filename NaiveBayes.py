import threading
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def logsumexp(Z):  # dimension C x N
    Zmax = np.max(Z, axis=0)[None, :]  # max over C
    log_sum_exp = Zmax + np.log(np.sum(np.exp(Z - Zmax), axis=0))
    return log_sum_exp


def evaluate_acc(y_test, y_pred):
    """
    Evaluates the accuracy of a model's prediction
    :param y_test: np.ndarray - the true labels
    :param y_pred: np.ndarray - the predicted labels
    :return: float - prediction accuracy
    """
    return np.sum(y_pred == y_test) / y_pred.shape[0]


class NaiveBayes:
    """
    Threaded implementation of the Naive Bayes ML algorithm using the Multinomial likelihood
    to classify text documents.
    """
    def __init__(self):
        pass

    def fit(self, x, y):
        """
        Fits the model to the given training data by learning the model parameters for each feature & computes the
        prior for each class.
        :param x: np.ndarray - the training data represented as a count matrix using CountVectorizer from sklearn
        :param y: np.ndarray - the labels corresponding to the training data (in the same order). Labels must start at
                               0 and end at (# of labels - 1). e.g. if there are 3 possible labels, they must be 0, 1, 2
        :return: the NaiveBayes model fitted on the given data
        """
        N, D = x.shape
        self.C = np.max(y) + 1  # stores the num of labels/categories
        probs = np.zeros((self.C, D))   # initializes the array that will store the model parameters
        Nc = np.zeros(self.C)  # initializes the array that will store the # of instances of each label/category

        # for each class get the MLE for each d,c (rel frequencies)
        for c in range(self.C):
            x_c = x[y == c]  # filter the rows of the data to all rows where label = c
            Nc[c] = x_c.shape[0]  # stores the number of instances where label = c

            num_threads = 10
            # calculates the interval of features for which each thread will work in
            interval_size = int(D / num_threads)
            threads = []    # stores the created threads
            for thread in range(num_threads):
                # creates threads
                threads.append(
                    threading.Thread(
                        target=NaiveBayes._fit_thread,
                        args=(x_c, c, thread*interval_size, (thread*interval_size) + interval_size, probs))
                )
                threads[thread].start()

            # waits for all threads to finish execution
            for thread in threads:
                thread.join()

            # without threading - this is the work being split up across n threads
            #for d in range(D):
                # count_d = np.sum(x_c[:, d]) + 1     # counts of word d in all documents labelled c
                # tot_count = np.sum(x_c)             # total word count in all documents labelled c
                # probs[c][d] = count_d / tot_count   # MLE for each d,c (rel frequency)

        self.probs = probs  # stores the learnt model parameters in self
        self.pi = (Nc + 1) / (N + self.C)  # stores the learnt priors using Laplace smoothing
        return self

    def _fit_thread(x_c, c, d_start, d_end, probs):
        """
        Thread fitting functionality, not meant to be used elsewhere.
        """
        for d in range(d_start, d_end):
            count_d = np.sum(x_c[:, d]) + 1  # counts of word d in all documents labelled c
            tot_count = np.sum(x_c)  # total word count in all documents labelled c
            probs[c][d] = count_d / tot_count  # MLE for each d,c (rel frequency)

    def predict(self, xt):
        """
        Predicts the labels of the given instances using the learnt parameters
        :param xt: np.ndarray - the test data represented as a count matrix using CountVectorizer from sklearn
        :return: np.ndarray - the posterior probabilities for each instance for each label. To get the model's label
                              prediction, simply take the label with the highest probability value for each instance.
        """
        Nt, D = xt.shape
        # for numerical stability we work in the log domain
        # we add a dimension because this is added to the log-likelihood matrix
        # that assigns a likelihood for each class (C) to each test point, and so it is C x N
        log_prior = np.log(self.pi)[:, None]

        # computes the Multinomial log likelihoods
        log_likelihood = np.zeros((Nt, self.C))
        for i in range(Nt):
            a = np.log(self.probs ** xt[i])
            b = np.sum(a, axis=1)
            log_likelihood[i] = b

        log_posterior = log_prior.T + log_likelihood
        posterior = np.exp(log_posterior - logsumexp(log_posterior))
        return posterior  # dimension N x C


if __name__ == '__main__':
    """
    selects the dataset to run the model on:
    1 -> 20Newsgroups dataset
    2 -> Sentiment140 dataset
    """
    dataset = 2

    # opens & stores the stop words as a list
    with open("stopwords.txt") as f:
        stop_words = f.read()
        stop_words = stop_words.split('\n')

    if dataset == 1:    # 20 news groups dataset
        twenty_train = fetch_20newsgroups(subset='train',
                                          shuffle=True,
                                          remove=(['headers', 'footers', 'quotes']))

        vectorizer = CountVectorizer(max_features=5000, stop_words=stop_words)
        X_counts = vectorizer.fit_transform(twenty_train.data)  # creates count matrix

        tfidf_transformer = TfidfTransformer()
        X_tfidf = tfidf_transformer.fit_transform(X_counts)

        # test-train split
        x_train, x_test, y_train, y_test = model_selection.train_test_split(X_tfidf, twenty_train.target,
                                                                            test_size=0.2)

        x_train, x_test = x_train.toarray(), x_test.toarray()

    else:   # sentiment140 dataset
        Sentiment140_test = pd.read_csv('data/testdata.manual.2009.06.14.csv', encoding='ISO-8859-1',
                                         header=None)
        Sentiment140_test = Sentiment140_test.loc[Sentiment140_test[0] != 2]    # removes instances with label = 2
        num_of_test_instances = Sentiment140_test.shape[0]
        Sentiment140 = pd.read_csv('data/training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1',
                                         header=None)

        # appends the train & test datasets together to vectorize them identically & will be split after
        Sentiment140 = Sentiment140.append(Sentiment140_test)

        Sentiment_columns = ['Y', 'id', 'date', 'query', 'user', 'text']
        Sentiment140.columns = Sentiment_columns

        # replaces all labels of 4 with 1, to respect the model's implementation requiring labels to be consecutive
        Sentiment140['Y'].replace({4: 1}, inplace=True)

        vectorizer = CountVectorizer(max_features=2000, stop_words=stop_words)
        X_counts = vectorizer.fit_transform(Sentiment140[['text']].values.flatten().tolist())

        tfidf_transformer = TfidfTransformer()
        X_tfidf = tfidf_transformer.fit_transform(X_counts).toarray()

        # splits the combined data back into test and train sets as they were given
        x_train = X_tfidf[:-num_of_test_instances, :]
        x_test = X_tfidf[-num_of_test_instances:, :]

        y_train = Sentiment140[['Y']].values[:-num_of_test_instances, :].flatten()
        y_test = Sentiment140[['Y']].values[-num_of_test_instances:, :].flatten()

    model = NaiveBayes()
    model.fit(x_train, y_train)
    y_prob = model.predict(x_test)
    y_pred = np.argmax(y_prob, axis=1)  # selects the label with the highest likelihood for each instance
    accuracy = evaluate_acc(y_test, y_pred)
    print(f"Accuracy is {accuracy}")

