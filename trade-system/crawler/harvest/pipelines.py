# Basic Scrapy settings:
# - respect robots.txt
# - slow down a bit so we don't hammer sites
# - send items to Postgres via our pipeline

BOT_NAME = "harvest"

ROBOTSTXT_OBEY = True
DOWNLOAD_DELAY = 1.0

ITEM_PIPELINES = {
    "harvest.pipelines.PostgresPipeline": 300
}

