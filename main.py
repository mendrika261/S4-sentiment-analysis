import time

from matplotlib import pyplot as plt
from nltk.corpus import wordnet
from sklearn.ensemble import RandomForestClassifier

from classification import *
from clustering import Kmeans

if __name__ == '__main__':
    start = time.time()

    # nltk.download('wordnet')

    text = "Feeling amazed by the capabilities of machine learning! It's revolutionizing industries like healthcare, " \
           "finance, and transportation, making processes more efficient and improving outcomes. " \
           "#ArtificialIntelligence #ML"

    limit = 18000
    model, words_idf = get_model_with_words_idf(limit, RandomForestClassifier, score=True)
    print(len(words_idf))
    print(prediction(model, words_idf, text))

    """limit = 250
    data = get_tf_idf_df(limit)
    x = data.iloc[:, 6:]
    print(Kmeans.get_best_k(x, min_k=1, max_k=20))"""

    """limit = 1000  # 1000:76s
    model, words_idf = get_model_with_words_idf(limit, Kmeans, random_state=0, k=12)
    # plt.imshow(prediction(model, words_idf, get_original_df(limit).iloc[12][5])[0][1], interpolation='bilinear')
    # plt.axis('off')
    # plt.show()
    model.show_word_cloud()
    print(model.data['k_means_tag'].value_counts())"""
    # print(model.data[model.data['k_means_tag'] == 0])"""

    """data = get_tf_idf_df(limit)
    x = data.iloc[:, 6:]
    modelK = Kmeans()
    modelK.fit(x, 10)
    print(x.iloc[0])
    print(prediction(modelK, words_idf, "love you guys"))"""

    end = time.time()
    print(f'\n\nTime: {round(end - start, 2)}s')
