# YouTubeCommentCrawler.py

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import time
from bs4 import BeautifulSoup

def collect_youtube_comments(search_keyword, max_videos=5):
    options = Options()
    options.add_argument("--headless")  # 창을 띄우지 않음
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920x1080")
    
    driver = webdriver.Chrome(options=options)

    try:
        # 1. 유튜브 접속 및 검색
        driver.get("https://www.youtube.com/")
        time.sleep(2)

        search_box = driver.find_element(By.NAME, "search_query")
        search_box.send_keys(search_keyword)
        search_box.send_keys(Keys.RETURN)
        time.sleep(2)

        # 2. 검색 결과에서 상위 max_videos개 영상 링크 수집
        video_links = []
        video_elems = driver.find_elements(By.ID, 'video-title')
        
        for elem in video_elems:
            link = elem.get_attribute('href')
            if link and 'watch' in link:
                video_links.append(link)
            if len(video_links) >= max_videos:
                break
        
        comments = []
        
        # 3. 각 영상 들어가서 댓글 수집
        for video_link in video_links:
            driver.get(video_link)
            time.sleep(3)

            # 스크롤 내려서 댓글 로드
            last_height = driver.execute_script("return document.documentElement.scrollHeight")
            for _ in range(5):  # 5번 스크롤 (조정 가능)
                driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
                time.sleep(2)
                new_height = driver.execute_script("return document.documentElement.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height

            # 페이지 소스 가져오기
            soup = BeautifulSoup(driver.page_source, "html.parser")

            # 댓글 요소 추출
            comment_elems = soup.select('ytd-comment-thread-renderer #content-text')
            for comment_elem in comment_elems:
                comment = comment_elem.text.strip()
                if comment:
                    comments.append(comment)

    finally:
        driver.quit()

    return comments
