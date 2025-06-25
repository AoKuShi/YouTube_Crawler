import streamlit as st
import pandas as pd
from lib.YouTubeCommentCrawler import collect_youtube_comments
from lib.myLangchainServiceForYC import LLMSentimentAnalyzer

# Streamlit 페이지 설정
st.set_page_config(
    layout="wide",
    page_title="YouTube 댓글 크롤러 & 감성 라벨링"
)

# 사이드바 UI
st.sidebar.title("유튜브 댓글 크롤러 & 감성 라벨링")
search_query    = st.sidebar.text_input("검색어 입력", "")
max_videos      = st.sidebar.number_input("제목당 조회할 영상 수", min_value=1, max_value=20, value=5)
server_endpoint = st.sidebar.text_input("LLM 서버 엔드포인트", "http://localhost:1234")
model_name      = st.sidebar.text_input("LLM 모델명", "google/gemma-3-4b")
save_csv        = st.sidebar.checkbox("CSV 저장 여부")

# 실행 버튼
if st.sidebar.button("수집 및 라벨링 실행") and search_query:
    # 1) 댓글 수집
    with st.spinner("댓글을 수집 중입니다..."):
        comments = collect_youtube_comments(search_query, max_videos=max_videos)
    st.success(f"총 {len(comments)}개의 댓글을 수집했습니다.")

    # 2) DataFrame 생성
    df = pd.DataFrame(comments, columns=["comment"])

    # 3) 감성 분석
    sa = LLMSentimentAnalyzer(server_endpoint, model_name)
    labels = []
    progress_bar = st.progress(0)
    for idx, comment in enumerate(df['comment']):
        label = sa.analyze_sentiment(comment)
        labels.append(label)
        progress_bar.progress((idx + 1) / len(df))
    df['label'] = labels

    # 4) 결과 출력
    st.subheader("수집된 댓글 및 감성 라벨")
    st.dataframe(df)

    # 5) CSV 저장
    if save_csv:
        filename = f"{search_query}_labeled_comments.csv"
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        st.success(f"CSV 파일이 저장되었습니다: {filename}")

else:
    st.warning("검색어를 입력한 후 '수집 및 라벨링 실행' 버튼을 눌러주세요.")
