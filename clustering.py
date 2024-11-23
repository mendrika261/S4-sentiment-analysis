import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from wordcloud import WordCloud


class Kmeans:
    def __init__(self, random_state=0, k=None):
        self.data = None
        self.k = k
        np.random.seed(random_state)

    def k_means_random(self):
        tags = pd.DataFrame(np.random.randint(0, self.k, self.data.shape[0]), columns=['k_means_tag'])
        self.data = self.data.reset_index(drop=True)
        self.data = pd.concat([self.data, tags], axis=1)

    def get_means(self):
        return self.data.groupby('k_means_tag').mean()

    @staticmethod
    def distance(row1, row2):
        return np.sqrt(np.sum((row1 - row2) ** 2))

    @staticmethod
    def min_distance(row, means):
        distances = []
        for m in range(means.shape[0]):
            distances.append(Kmeans.distance(means.iloc[m], row))
        return np.argmin(np.array([distances]))

    def get_evaluation(self):
        means = self.get_means()
        return round(self.data.apply(lambda x: Kmeans.distance(x, means.loc[x['k_means_tag']]), axis=1).sum(), 0)

    @staticmethod
    def get_evaluations(x_train, max_k=10, min_k=2):
        result = {}
        for i in range(min_k, max_k+1):
            test = Kmeans(k=i)
            test.fit(x_train)
            result[i] = test.get_evaluation()
            print(f'k={i}: {test.get_evaluation()}')
        return result

    @staticmethod
    def get_best_k(x_train, max_k=10, min_k=2):
        evaluations = Kmeans.get_evaluations(x_train, max_k, min_k)

        # plot evaluations
        plt.plot(list(evaluations.keys()), list(evaluations.values()))
        plt.xlabel('k')
        plt.ylabel('evaluation')
        plt.xticks(np.arange(min_k, max_k+1, 1.0))
        plt.show()

        max_diff = float('-inf')
        best_k = min_k
        for i in range(min_k, max_k):
            diff = abs(evaluations.get(i) - evaluations.get(i+1))
            # print(diff)
            if diff > max_diff:
                # print(f'k: {i}')
                max_diff = diff
                best_k = i
        return best_k + 1

    def fit(self, x_train, **kwargs):
        self.data = x_train
        if self.k is None:
            self.k = Kmeans.get_best_k(x_train)

        self.k_means_random()
        print(self.data['k_means_tag'].value_counts())

        prev_means = []
        curr_means = self.get_means()
        print(curr_means)
        while str(prev_means) != str(curr_means):
            prev_means = curr_means
            self.data['k_means_tag'] = self.data.apply(lambda row: Kmeans.min_distance(row, curr_means),
                                                       axis=1)
            curr_means = self.get_means()
        return self

    def predict(self, value):
        means = self.get_means()
        result = []
        for i in range(value.shape[0]):
            temp = Kmeans.min_distance(value.iloc[i], means)
            result.append((temp, WordCloud().generate_from_frequencies(means.iloc[temp])))
        return result

    def show_word_cloud(self):
        means = self.get_means()
        for i in range(means.shape[0]):
            wordcloud = WordCloud().generate_from_frequencies(means.iloc[i])

            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.show()
