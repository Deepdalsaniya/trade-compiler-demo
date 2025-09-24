# A very simple spider:
# - starts from a few official sites
# - downloads pages
# - strips them to plain text
# - saves into Postgres through the pipeline

import scrapy
from bs4 import BeautifulSoup

START_URLS = [
  "https://www.cbp.gov/trade/rulings",
  "https://www.bis.doc.gov/index.php/all-articles"
]

class GovBasicSpider(scrapy.Spider):
    name = "gov_basic"
    allowed_domains = ["cbp.gov","bis.doc.gov"]
    start_urls = START_URLS

    def parse(self, response):
        # Turn HTML into clean text
        soup = BeautifulSoup(response.text, "html.parser")
        title = (soup.title.text.strip() if soup.title else response.url)
        text = " ".join(soup.get_text(" ").split())

        # Send item to the pipeline (which writes to Postgres)
        yield {"source": self.name, "url": response.url, "title": title, "text": text}

        # Follow links within the same allowed domains
        for a in soup.select("a[href]"):
            href = response.urljoin(a["href"])
            if any(d in href for d in self.allowed_domains):
                yield scrapy.Request(href, callback=self.parse)

