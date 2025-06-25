프로젝트 명 : YouTube Crawler![image](https://github.com/user-attachments/assets/eace8a99-2aec-49f0-96af-0012be61ac6f)



기존의 유트브 댓글 크롤러(YoutubeCrawlerForData.py)에 myLangchainService.py(교수님께 제공받은 LLM감성분석 코드)를 적용 시켜 댓글 크롤링과 라벨링 작업을 동시에 진행하는 코드(YoutubeCrawlerForDataV2.py)를 작성
myLangchainService.py는 LLM모델명과 엔드포인트, 프롬프트 수정 후 적용
프롬프트 수정 1 : 긍정, 부정, 중립 라벨링을 정수형(1, -1, 0)으로 전환
프롬프트 수정 2 : 토큰 수를 줄이기 위해 영문 프롬포트로 작성 
