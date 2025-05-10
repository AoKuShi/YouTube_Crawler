import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Okt

class SentimentAnalyzer:
    def __init__(self, tokenizer_file, sa_model_file):
        self.tokenizer = joblib.load(tokenizer_file)
        self.model = load_model(sa_model_file)
        self.ktokenizer = Okt().morphs
        self.label_map = ['부정', '중립', '긍정']
        self.max_len = self.model.input_shape[1]

    def sentiment_analysis(self, review):
        morphs = self.ktokenizer(review)  # 형태소 분석
        sequences = self.tokenizer.texts_to_sequences([morphs])  # 정수 인코딩
        X = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')  # 패딩
        preds = self.model.predict(X)
        max_index = np.argmax(preds[0])  # 가장 높은 확률 클래스 인덱스
        return self.label_map[max_index], float(preds[0][max_index])

sa_tokenizer_file = "./lib/model/product_sa_tokenizer.pkl"
sa_model_file = "./lib/model/product_sa_model_lstm.keras"
sa = SentimentAnalyzer(sa_tokenizer_file, sa_model_file)