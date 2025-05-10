1. AIHub 사이트에서 부정, 중립, 긍정(-1, 0, 1) 3가지로 라벨링된 감정분석 데이터(속성기반 감정분석 데이터/62mb)를 다운로드

2. 태깅이 완료된 모든 파일(json)을 통합, createCSV.py를 사용해 csv파일로 변환

3. LSTM으로 감성 분석 모델 생성, 저장(train_model.py)

4. 모델을 불러와 텍스트 감성 분석 함수, sentiment_analysis를 작성(sentiment_predictor.py)

5. selenium과 BeautifulSoup을 이용한 유튜브 크롤러 함수 collect_youtube_comments를 작성(YouTubeCommentCrawler.py)

6. collect_youtube_comments로 유튜브를 크롤링(키워드 검색)해 막대그래프와 워드 클라우드, 그리고 sentiment_analysis로 감성 분석의 평균치를 표기 하는 코드, YoutubeCrawler.py를 작성

7. streamlit run 명령어로 YoutubeCrawler.py를 실행
(이하 상세는 SRS, UCDiagram에 기술)
