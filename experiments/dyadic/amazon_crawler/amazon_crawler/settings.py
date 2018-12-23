# -*- coding: utf-8 -*-

# Scrapy settings for amazon_crawler project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     http://doc.scrapy.org/en/latest/topics/settings.html
#     http://scrapy.readthedocs.org/en/latest/topics/downloader-middleware.html
#     http://scrapy.readthedocs.org/en/latest/topics/spider-middleware.html

BOT_NAME = 'amazon_crawler'

SPIDER_MODULES = ['amazon_crawler.spiders']
NEWSPIDER_MODULE = 'amazon_crawler.spiders'

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# The S3 path to store items
ITEMS_STORE = "http://localhost:9000/monomer"
# The S3 path to store images
IMAGES_STORE = "http://localhost:9000/images"
# AWS S3 Keys
AWS_ACCESS_KEY_ID = "2PEO7HAJ1LU462J08EPZ"
AWS_SECRET_ACCESS_KEY = "lEW2oO8uGiIDPyoyvXYnJ2zxw5zwaWfTOVgXcHCG"
