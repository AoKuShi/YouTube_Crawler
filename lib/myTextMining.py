from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from wordcloud import WordCloud

def load_corpus_from_csv(corpus_file, col_name):
    data_df = pd.read_csv(corpus_file)

    # NaN 제거 + 문자열 아닌 데이터 제거
    result_series = data_df[col_name].dropna()
    result_series = result_series[result_series.apply(lambda x: isinstance(x, str))]

    return list(result_series)

def tokenize_korean_corpus(corpus_list, tokenizer, tags, stopwords):
    text_pos_list = []
    for text in corpus_list:
        text_pos = tokenizer(text)
        text_pos_list.extend(text_pos)

    token_list = [token for token, tag in text_pos_list if tag in tags and token not in stopwords]
    return token_list

def analyze_word_freq(corpus_list, tokenizer, tags, stopwords):
    token_list = tokenize_korean_corpus(corpus_list, tokenizer, tags, stopwords)
    counter = Counter(token_list)
    return counter

def visualize_barchart(counter, title, xlabel, ylabel):
    word_list = [word for word, count in counter.most_common(20)]
    count_list = [count for word, count in counter.most_common(20)]

    # matplotlib 한글 폰트 설정
    font_path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)

    # 수평 막대그래프
    plt.barh(word_list[::-1], count_list[::-1])

    # 그래프 정보 추가 
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def visualize_wordcloud(counter):
    font_path = "c:/Windows/fonts/malgun.ttf"

    wordcloud = WordCloud(
        font_path=font_path,
        width=1080,
        height=1080,
        max_words=100,
        background_color='ivory'
    )

    wordcloud = wordcloud.generate_from_frequencies(counter)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
