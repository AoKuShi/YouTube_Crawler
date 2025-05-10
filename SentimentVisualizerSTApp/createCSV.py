import os
import json
import csv

# 경로 설정
input_dir = 'lib/data/labelingData'
output_file = 'lib/data/sentiment_data.csv'

# 결과 저장용 리스트
rows = []

# 디렉터리 내 모든 JSON 파일 처리
for filename in os.listdir(input_dir):
    if filename.endswith('.json'):
        filepath = os.path.join(input_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                for entry in data:
                    raw_text = entry.get('RawText', '')
                    general_polarity = entry.get('GeneralPolarity', '')
                    aspects = entry.get('Aspects', [])
                    for aspect in aspects:
                        sentiment_text = aspect.get('SentimentText', '')
                        sentiment_polarity = aspect.get('SentimentPolarity', '')
                        rows.append([
                            raw_text.strip(),
                            general_polarity,
                            sentiment_text.strip(),
                            sentiment_polarity
                        ])
            except json.JSONDecodeError:
                print(f"⚠️ JSON 파싱 오류: {filename}")

# CSV 저장
with open(output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['RawText', 'GeneralPolarity', 'SentimentText', 'SentimentPolarity'])  # 헤더
    writer.writerows(rows)

print(f"완료: 총 {len(rows)}개의 데이터가 {output_file}에 저장되었습니다.")
