import streamlit as st
import pandas as pd
from myLangchainService import LLMSentimentAnalyzer, load_corpus_from_csv

server_endpoint="http://112.76.103.124:1234/v1"  # LM Studio 주소 (교수실)
#server_endpoint="http://127.0.0.1:1234/v1" # 집
model = "google/gemma-3-12b"

st.set_page_config(
    page_title="Sentiment Label Generator using LLM",
    page_icon="📊",
    layout="wide"
    #initial_sidebar_state="expanded"
)

@st.dialog("데이터 확인하기", width='large')
def view_raw_data_dialog(data_df):
    num_data = st.number_input("확인할 데이터 수", value=10)
    st.write(data_df.head(num_data))

with st.sidebar:
    data_file = st.file_uploader("파일 선택", type=['csv'])
    column_name = st.text_input('데이터가 있는 컬럼명', value='review')
    if st.button("데이터 파일 확인"): 
        if data_file:
            data_df = pd.read_csv(data_file)
            view_raw_data_dialog(data_df)
            st.info(f'데이터 수 : {len(data_df)}')
        else:
            st.sidebar.warning("데이터 파일을 업로드 후 데이터를 확인하세요.")
    
    with st.form('my_form'):
        st.write("## 데이터 범위")
        start = st.number_input("레이블링 시작 문서 index", 0)
        numData = st.number_input("레이블링 문서 수", value=100)
        st.write("## LLM 설정")
        server_endpoint = st.text_input("LLM server endpoint", value=server_endpoint)
        model = st.text_input("LLM model", value=model)
        submitted = st.form_submit_button('레이블링 시작')

st.title("💬 Sentiment Label Generator using LLM")
status = st.info("업로드한 파일의 " + column_name + "에 대한 감성(긍정, 중립, 부정) 레이블링을 진행합니다.")

def start_sentiment_label(target, start, end, display_result=print, display_status=print):
    sa = LLMSentimentAnalyzer(server_endpoint, model)
    label_list = []
    count = 0
    for i in range(start, end):
        review = target[i]
        count += 1
        response = sa.analyze_sentiment(review)
        display_result(f'[{response}]{review}')
        label_list.append(response)
        if count % 50 == 0:
             display_status(f'{count} 개의 문서를 레이블링했습니다.')
    return label_list


if submitted:
    if not data_file:
        st.error('레이블링 대상 파일을 업로드 한 후 분석 시작하세요.')
        exit()

    status.info('레이블링을 시작합니다.')

    corpus = load_corpus_from_csv(data_file, column_name)
    if not corpus:
        st.error(f"분석할 컬럼명 '{column_name}'을 확인하고 다시 입력해주세요.")
        exit()

    c = st.container()

    label_list = start_sentiment_label(corpus, start, start+numData, c.write, status.info)

    label_df = pd.DataFrame({'document' : corpus[start:start+numData], 'label' : label_list})
    label_df.to_csv(f'new_sentiment_{start}_{start+numData-1}.csv', index=False)
