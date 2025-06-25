# YouTube Crawler

---
## 개요 : YouTube 댓글 감성 분석 웹 프로젝트
  - 사용자 입력 키워드로 YouTube 댓글을 자동 수집하고, LLM 기반 감성 분석을 통해 긍정·부정·중립으로 라벨링한 뒤, 웹 인터페이스에서 시각화 및 통계 제공
  - 과정 : 데이터 준비 및 전처리 → 모델 학습 및 저장 → YouTube 댓글 크롤링 → Streamlit 웹 애플리케이션
  - python 인터프리터 설정 : requirements.txt 참고

---
## 데이터 준비 및 전처리
  - 기존의 유트브 댓글 크롤러에 LLM_Labeling/myLangchainService.py(LLM감성분석 코드)를 적용 시켜 댓글 크롤링과 라벨링 작업을 동시에 진행하는 코드(YoutubeCrawlerForDataV2.py)를 작성
  - LLM_Labeling/myLangchainService.py는 LLM모델명과 엔드포인트, 프롬프트 수정(lib/myLangchainServiceForYC.py) 후 적용 

### <기존 LLM감성 분석 페이지(LLM_Labeling/SentimentLabelingSTApp.py)>
  - ![image](https://github.com/user-attachments/assets/bebc4f9f-e677-4fe0-b864-fe034cf5258c)
  - 크롤링 후 엑셀파일을 업로드해 작업을 진행(크롤링 따로 감성분석 라벨링 따로)

### <새로운 LLM감성 분석 페이지(YoutubeCrawlerForDataV2.py)>
  - ![image](https://github.com/user-attachments/assets/9487180a-2f73-4704-96b1-8a33e027e847)
  - 크롤링과 감성분석을 동시에 진행

### <gemma-3-4b LLM모델 사용>
  - ![image](https://github.com/user-attachments/assets/dc0d81fa-1c37-4550-ad0f-a14865c29f7b)

### <LLM모델 라벨링 프롬프트>
  - ![image](https://github.com/user-attachments/assets/3d6445ae-b5d4-4182-8626-4e33ac88e6fa)
    - 프롬프트 수정 1 : 긍정, 부정, 중립 라벨링을 정수형(1, -1, 0)으로 전환
    - 프롬프트 수정 2 : 토큰 수를 줄이기 위해 영문 프롬포트로 작성

### <최종 라벨링 데이터>
  - ![image](https://github.com/user-attachments/assets/0f87bdb8-9db6-4379-9bc7-cf459add5017)

---
## 모델 학습 및 저장
  - train_model.py로 LSTM 모델 학습
    - Embedding → LSTM → Dense
  - EarlyStopping 적용 후 최적 모델 저장
  - Tokenizer와 모델을 lib/model 폴더에 저장

### <훈련 과정>
  - ![스크린샷 2025-06-25 152417](https://github.com/user-attachments/assets/9089413f-561e-4f81-a387-e0ebfdd9b2a1)
  - ![스크린샷 2025-06-25 152426](https://github.com/user-attachments/assets/fb42bdc6-a6ef-463d-88b5-91f712928a64)
  - 클래스별 특징
    - 긍정: Precision·Recall 모두 높아(0.96/0.98), 안정적
    - 부정: Recall이 0.93으로 높으나 Precision 0.89라 소수의 오탐 있음
    - 중립: Recall이 0.46으로 매우 낮아(절반 이상이 다른 클래스로 분류됨), F1도 0.58에 그침
  - 종합 평가
    - 과적합 초기 신호: 검증 손실과 정확도가 1~2 에포크 이후 하락세
    - 중립 클래스 약함: 중립 샘플이 부정·긍정으로 많이 오분류
  - 개선 방안
    - 조기 종료 시점 조정: patience를 1–2로 줄여 과적합 방지
    - 클래스 가중치(class_weight) 부여 또는 오버샘플링으로 중립 데이터 비율 높이기
    - 드롭아웃, L2 정규화 추가로 모델 일반화 강화

### <그래프>
  - ![download01](https://github.com/user-attachments/assets/d63a9106-dfdf-4702-94bd-bd9c18bcc589)
  - ![download02](https://github.com/user-attachments/assets/96b959fb-eb2e-45a7-9015-fd84a79ea9ee)
  - ![458728646-c429fed8-2840-4010-84e7-fbc2200de48f](https://github.com/user-attachments/assets/deeed725-5f57-4f93-b63c-d33926efb869)
  - 해석 : 모델이 초반에는 검증 데이터에 잘 맞춰 학습하다가, 이후 훈련 데이터에 과도하게 적합(overfitting)되어 검증 성능이 하락한 것으로 보임

---
## YouTube 댓글 크롤링(YouTubeCommentCrawler.py)
  - 유튜브 클롤링 함수, collect_youtube_comments 작성
    - selenium + BeautifulSoup 기반
  - 검색어로 상위 10개 영상 댓글 자동 수집
  - 댓글 내용을 리스트로 반환

---
## Streamlit 웹 애플리케이션(YoutubeCrawler.py)
  - 검색어 입력, CSV 저장/업로드 UI 구현
  - 키워드 빈도그래프, 워드클라우드 시각화
  - sentiment_predictor로 감성 분석 후 평균 확률 테이블 출력(sentiment_predictor.py)

---
### <분석 결과 예시 - 긍정>
  - ![image](https://github.com/user-attachments/assets/106608e1-269a-4910-9b9e-7e68047e1d70)

### <분석 결과 예시 - 부정>
  - ![image](https://github.com/user-attachments/assets/44eab064-77eb-4997-b07d-df6885c00a19)

### <유스케이스 다이어그램>
  - ![image](https://github.com/user-attachments/assets/363218dd-eca8-4112-a8a0-189a594618f6)

### <요구사항 명세서>
  - ![image](https://github.com/user-attachments/assets/f0db87c7-54a3-42b2-b9ce-5eb469ecce8f)

---
# 테크리포트(요약)
  - ## 목적  
    - 키워드별 YouTube 댓글 자동 수집 및 감성(긍정·부정·중립) 분석  
  - ## 흐름  
    - 댓글 크롤링 + LLM 라벨링 → LSTM 모델 학습/예측 → Streamlit UI 시각화  
  - ## 주요 기술  
    - Selenium/BeautifulSoup, LangChain LLM, TensorFlow Keras(LSTM), Streamlit  

- ## 성과  
  - **댓글 200개 수집**: 약 3초  
  - **LLM 라벨링 정확도**: 85%  
  - **LSTM 모델 전체 정확도**: **94%**  
    - 클래스별 리콜:  
      - 부정 (Recall) ≈ 93%  
      - 중립 (Recall) ≈ 46%  
      - 긍정 (Recall) ≈ 98%  

- ## 향후 과제  
  1. 크롤링 단계에서 GPU 활용으로 속도 향상  
  2. 대시보드 고도화  
     - 검색할 영상 갯수, 댓글 수, 분석 카테고리 등 사용자별 커스터마이징 지원
     - BERT 전이학습 적용을 통한 모델 정확도 향상
