import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from konlpy.tag import Okt
import joblib
from tqdm import tqdm

# 1. 데이터 로딩
df = pd.read_csv("./lib/data/sentiment_data.csv")
df.dropna(inplace=True)

# 2. 형태소 분석 (진행도 표시)
okt = Okt()
tqdm.pandas(desc="형태소 분석 중")
df['Tokenized'] = df['SentimentText'].progress_apply(lambda x: okt.morphs(x))

# 3. 정수 인코딩 및 패딩
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['Tokenized'])
sequences = tokenizer.texts_to_sequences(df['Tokenized'])

max_len = 45
X = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

# 4. 라벨 전처리: -1 → 0, 0 → 1, 1 → 2
y = np.array([label + 1 for label in df['SentimentPolarity'].values])
y_cat = to_categorical(y, num_classes=3)

# 5. 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# 6. 모델 구성
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
    LSTM(128),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')  # 3클래스 (부정/중립/긍정)
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 7. 모델 학습
es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test), callbacks=[es])

# 8. 모델 평가
preds = model.predict(X_test)
pred_labels = np.argmax(preds, axis=1)
true_labels = np.argmax(y_test, axis=1)

label_map = ['부정', '중립', '긍정']
print(classification_report(true_labels, pred_labels, target_names=label_map))

# 9. 모델 및 토크나이저 저장
save_dir = "./lib/model"
os.makedirs(save_dir, exist_ok=True)

tokenizer_path = os.path.join(save_dir, "product_sa_tokenizer.pkl")
model_path = os.path.join(save_dir, "product_sa_model_lstm.keras")

joblib.dump(tokenizer, tokenizer_path)
model.save(model_path)

print(f"\n모델 저장 완료:\n- Tokenizer: {tokenizer_path}\n- 모델: {model_path}")
