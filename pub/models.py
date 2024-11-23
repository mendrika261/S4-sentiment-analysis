import base64
from io import BytesIO

from django.db import models
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from classification import prediction, get_model_with_words_idf
from clustering import Kmeans

model_sentiment_limit = 18000
model_category_limit = 1000
sentiment, words_idf_sent = get_model_with_words_idf(model_sentiment_limit, RandomForestClassifier)
category, words_idf_cat = get_model_with_words_idf(model_category_limit, Kmeans, random_state=0, k=12)


class Comment(models.Model):
    content = models.CharField(max_length=1000)

    def is_positive(self):
        if int(prediction(sentiment, words_idf_sent, self.content)[0]) == 0:
            return False
        return True

    def about(self):
        wordcloud = prediction(category, words_idf_cat, self.content)[0][1]
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return {"image": image_data}
