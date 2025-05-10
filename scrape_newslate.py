from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from tqdm import tqdm
import time
import json

BASE_URL = "https://newslate.co.uk"
visited = set()
pages = []

# Setup Selenium
def get_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--window-size=1920,1080")

    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=chrome_options)


def is_internal(url):
    parsed_url = urlparse(urljoin(BASE_URL, url))
    return parsed_url.netloc == urlparse(BASE_URL).netloc

def normalize_url(url):
    url = urljoin(BASE_URL, url)
    parsed = urlparse(url)
    return parsed.scheme + "://" + parsed.netloc + parsed.path.rstrip('/')

def extract_text_from_html(soup):
    content = []
    for tag in soup.find_all(['p', 'li', 'span', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        text = tag.get_text(separator=' ', strip=True)
        if text:
            content.append(text)
    return "\n".join(content)

def scrape_page(driver, url):
    try:
        driver.get(url)
        time.sleep(2)  # Wait for JS to render
        soup = BeautifulSoup(driver.page_source, "html.parser")
        text = extract_text_from_html(soup)
        return soup, text
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return None, None

def crawl_site(start_url, max_pages=100):
    to_visit = [start_url]
    driver = get_driver()

    with tqdm(total=max_pages, desc="Scraping site") as pbar:
        while to_visit and len(visited) < max_pages:
            current = normalize_url(to_visit.pop(0))
            if current in visited:
                continue
            visited.add(current)

            soup, text = scrape_page(driver, current)
            if soup is None or not text:
                continue

            pages.append({
                "url": current,
                "text": text
            })
            pbar.update(1)

            for link_tag in soup.find_all("a", href=True):
                href = link_tag['href']
                full_url = normalize_url(href)
                if is_internal(full_url) and full_url not in visited:
                    to_visit.append(full_url)

            time.sleep(1)  # Politeness delay

    driver.quit()
    return pages

if __name__ == "__main__":
    results = crawl_site(BASE_URL, max_pages=100)

    with open("python_backend\newslate_scraped_content.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Scraped {len(results)} pages from {BASE_URL}")