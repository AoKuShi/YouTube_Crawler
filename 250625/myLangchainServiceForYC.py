from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import pandas as pd

class LLMSentimentAnalyzer:
    def __init__(self, server_endpoint, model):
        # server_endpoint 예시: "http://localhost:1234"
        # → 여기에 "/v1"을 붙여서 올바른 base_url 생성
        api_base = server_endpoint.rstrip("/") + "/v1"

        self.system_message = (
            "You are a world-class sentiment analysis specialist."
            "When given a sentence, classify its sentiment as follows: Return 1 if the sentiment is positive; return -1 if the sentiment is negative; return 0 if the sentiment is neutral."
            "Answer with exactly one of the three integers (1, -1, or 0) and nothing else—no explanations or additional text."
        )
        self.human_message = "다음 문장을 분석해 주세요: {input_sentence}"

        # ↓ 변경된 부분: base_url 에 반드시 /v1 포함
        self.llm = ChatOpenAI(
            base_url=api_base,
            api_key="not needed",  # 로컬 서버라면 실제 키가 없어도 무방
            model=model
        )

        self.template = ChatPromptTemplate.from_messages([
            ("system", self.system_message),
            ("human", self.human_message)
        ])
        self.parser = StrOutputParser()
        self.chain = self.template | self.llm | self.parser

    def analyze_sentiment(self, sentence):
        return self.chain.invoke({"input_sentence": sentence})


def load_corpus_from_csv(filename, column):
    data_df = pd.read_csv(filename)
    if column in data_df.columns:
        data_df = data_df.dropna(subset=[column])
        return list(data_df[column])
    return None
