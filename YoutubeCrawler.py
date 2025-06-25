import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from matplotlib import rc, font_manager
from konlpy.tag import Okt

import lib.myTextMining as tm
import lib.YouTubeCommentCrawler as yc

from sentiment_predictor import SentimentAnalyzer
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 감성 분석 모델 로드
sa_tokenizer_file = "./lib/model/product_sa_tokenizer.pkl"
sa_model_file     = "./lib/model/product_sa_model_lstm.keras"
sa = SentimentAnalyzer(sa_tokenizer_file, sa_model_file)

# 한글 폰트 설정
font_path = "c:/Windows/Fonts/malgun.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# Streamlit 설정
st.set_page_config(layout="wide", page_title="YouTube 댓글 감성 분석")

# 사이드바
st.sidebar.title("유튜브 댓글 분석")
search_query = st.sidebar.text_input("검색어 입력", "")
save_csv     = st.sidebar.checkbox("CSV 저장 여부")
uploaded_file= st.sidebar.file_uploader("CSV 파일 업로드", type=["csv"])

# 슬라이더
num_words_freq = st.sidebar.slider("빈도수 그래프 단어 수", 5, 50, 20)
num_words_wc   = st.sidebar.slider("워드클라우드 단어 수", 10,100, 50)

# 데이터 로딩
df = None
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("CSV 파일 업로드 완료")

elif st.sidebar.button("댓글 수집 실행") and search_query:
    comments = yc.collect_youtube_comments(search_query, max_videos=5)
    df = pd.DataFrame(comments, columns=['comment'])
    if save_csv:
        filename = f"./lib/data/{search_query}_youtube_comments.csv"
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        st.sidebar.success("CSV 저장 완료")

# 키워드 시각화
if df is not None and "comment" in df.columns:
    st.subheader("댓글 키워드 분석 결과")
    corpus_list = df["comment"].dropna().tolist()

    # 형태소 분석 및 전처리
    okt = Okt()
    stopwords = ["이","그","저","것","수","더","때","가","을","를","은","는","에","의","와","과","도","한","하다","댓글","영상","편집"]
    tags = ['Noun']
    token_list = tm.tokenize_korean_corpus(corpus_list, okt.pos, tags, stopwords)
    counter = Counter(token_list)

    # 빈도수 그래프
    top_words_freq = counter.most_common(num_words_freq)
    words_freq, counts_freq = zip(*top_words_freq)
    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(x=list(counts_freq), y=list(words_freq), ax=ax, palette="coolwarm")
    ax.set_title("댓글 키워드 빈도수")
    ax.set_xlabel("빈도수"); ax.set_ylabel("단어")
    st.pyplot(fig)

    # 워드클라우드
    top_words_wc = dict(counter.most_common(num_words_wc))
    wordcloud = WordCloud(font_path=font_path, width=800, height=400, background_color="white")\
                .generate_from_frequencies(top_words_wc)
    fig_wc, ax_wc = plt.subplots(figsize=(8,4))
    ax_wc.imshow(wordcloud, interpolation="bilinear")
    ax_wc.axis("off")
    st.pyplot(fig_wc)

    # 4. 댓글 감성 분석
    st.subheader("댓글 감성 분석 (전체 댓글 기준)")

    comments = df["comment"].dropna().tolist()
    # 1) 형태소 분석 → 정수 인코딩 → 패딩
    seqs = [sa.ktokenizer(c) for c in comments]
    seqs = sa.tokenizer.texts_to_sequences(seqs)
    X_comments = pad_sequences(seqs, maxlen=sa.max_len, padding='post', truncating='post')

    # 2) 예측 → 평균 확률 계산
    preds = sa.model.predict(X_comments)        # shape=(n_comments, 3)
    mean_probs = preds.mean(axis=0)             # [p_neg, p_neu, p_pos]
    overall = sa.label_map[np.argmax(mean_probs)]

    # 3) 결과 출력
    st.write(f"**전체 댓글 감성:** {overall}")
    df_sent = pd.DataFrame({
        "감성": sa.label_map,
        "평균 확률": np.round(mean_probs, 3)
    })
    st.table(df_sent)
    # ─────────────────────────────────────────────

else:
    st.warning("검색어 입력 후 댓글 수집을 실행하거나 CSV 파일을 업로드하세요.")
