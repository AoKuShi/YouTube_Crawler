# YouTube Crawler
---
개요 : YouTube 댓글 감성 분석 웹 프로젝트
  - 사용자 입력 키워드로 YouTube 댓글을 자동 수집하고, LLM 기반 감성 분석을 통해 긍정·부정·중립으로 라벨링한 뒤, 웹 인터페이스에서 시각화 및 통계 제공
  - python 인터프리터 설정 : requirements.txt 참고
---
데이터 준비 및 전처리
  - 기존의 유트브 댓글 크롤러(YoutubeCrawlerForData.py)에 myLangchainService.py(교수님께 제공받은 LLM감성분석 코드)를 적용 시켜 댓글 크롤링과 라벨링 작업을 동시에 진행하는 코드(YoutubeCrawlerForDataV2.py)를 작성
  - myLangchainService.py는 LLM모델명과 엔드포인트, 프롬프트 수정 후 적용
      - 프롬프트 수정 1 : 긍정, 부정, 중립 라벨링을 정수형(1, -1, 0)으로 전환
      - 프롬프트 수정 2 : 토큰 수를 줄이기 위해 영문 프롬포트로 작성 

<SentimentLabelingSTApp.py 실행>

![image](https://github.com/user-attachments/assets/bebc4f9f-e677-4fe0-b864-fe034cf5258c)

<gemma-3-4b LLM모델 사용>

![image](https://github.com/user-attachments/assets/dc0d81fa-1c37-4550-ad0f-a14865c29f7b)

<LLM모델 라벨링 프롬프트>

![image](https://github.com/user-attachments/assets/3d6445ae-b5d4-4182-8626-4e33ac88e6fa)

<YoutubeCrawlerForDataV2.py 실행>

![image](https://github.com/user-attachments/assets/9487180a-2f73-4704-96b1-8a33e027e847)

<최종 라벨링 데이터>

![image](https://github.com/user-attachments/assets/0f87bdb8-9db6-4379-9bc7-cf459add5017)
---
모델 학습 및 저장
  - train_model.py로 LSTM 모델 학습
    - Embedding → LSTM → Dense
  - EarlyStopping 적용 후 최적 모델 저장
  - Tokenizer와 모델을 lib/model 폴더에 저장
---
YouTube 댓글 크롤링
  - 유튜브 클롤링 함수, collect_youtube_comments 작성
    - selenium + BeautifulSoup 기반
  - 검색어로 상위 10개 영상 댓글 자동 수집
  - 댓글 내용을 리스트로 반환
---
Streamlit 웹 애플리케이션
  - 검색어 입력, CSV 저장/업로드 UI 구현
  - 키워드 빈도그래프, 워드클라우드 시각화
  - sentiment_predictor로 감성 분석 후 평균 확률 테이블 출력
---
<분석 결과 예시 - 긍정>

![image](https://github.com/user-attachments/assets/106608e1-269a-4910-9b9e-7e68047e1d70)

<분석 결과 예시 - 부정>

![image](https://github.com/user-attachments/assets/44eab064-77eb-4997-b07d-df6885c00a19)

<유스케이스 다이어그램>

![image](https://github.com/user-attachments/assets/363218dd-eca8-4112-a8a0-189a594618f6)

<요구사항 명세서>

![image](https://github.com/user-attachments/assets/f0db87c7-54a3-42b2-b9ce-5eb469ecce8f)

---
# 테크리포트(요약)
  - ## 목적: 키워드별 YouTube 댓글 자동 수집 및 감성(긍정·부정·중립) 분석
  - ## 흐름: 댓글 크롤링 + LLM 라벨링 → LSTM 모델 학습/예측 → Streamlit UI 시각화
  - ## 주요 기술: Selenium/BeautifulSoup, Langchain LLM, TensorFlow Keras(LSTM), Streamlit
  - ## 성과: 댓글 200개 수집 3초, LLM 라벨링 정확도 85%, 모델 정확도 88%
  - ## 향후 과제: 크롤링 시 GPU 사용으로 속도up, 대시보드 고도화(검색할 영상 갯수, 댓글 수, 검색 카테고리 등 커스터마이징이 가능하게 변경)
