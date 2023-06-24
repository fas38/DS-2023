import scrapy
from scrapy.crawler import CrawlerProcess
import json


class StockSpider(scrapy.Spider):
    name = 'stock_spider'
    start_urls = [
        'https://stockanalysis.com/stocks/industry/internet-content-and-information/',
        'https://stockanalysis.com/stocks/industry/entertainment/',
        'https://stockanalysis.com/stocks/industry/travel-services/'
    ]

    custom_settings = {
        'FEED_FORMAT': 'json',
        'FEED_URI': 'stocks.json'
    }

    def parse(self, response):
        rows = response.css('table tbody tr')
        for row in rows:
            yield {
                'symbol': row.css('td:nth-child(2) a::text').get(),
                'company_name': row.css('td:nth-child(3)::text').get(),
                'market_cap': row.css('td:nth-child(4)::text').get(),
                'percent_change': row.css('td:nth-child(5)::text').get(),
                'volume': row.css('td:nth-child(6)::text').get(),
                'revenue': row.css('td:nth-child(7)::text').get(),
            }

if __name__ == "__main__":
    process = CrawlerProcess(settings={
        "FEEDS": {
            "stocks.json": {"format": "json"},
        },
    })

    process.crawl(StockSpider)
    process.start()  # the script will block here until the crawling is finished
