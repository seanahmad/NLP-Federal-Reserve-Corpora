# System:
import sys
import os
import re
from datetime import date
from datetime import datetime

# Computation:
import numpy as np
import pandas as pd
import pickle

# Web Scraping:
import json
from bs4 import BeautifulSoup
from tqdm import tqdm
import requests
import threading
from abc import ABCMeta, abstractmethod
print(sys.stdout.encoding)

# Text Extraction:
# Tika depends on Java version, so use textract instead as the pdf is anyway a simple text only
# # User TIKA for pdf parsing
# os.environ['TIKA_SERVER_JAR'] = 'https://repo1.maven.org/maven2/org/apache/tika/tika-server/1.19/tika-server-1.19.jar'
# import tika
# from tika import parser
import textract

# Import parent class
from fomc_get_data.FomcBase import FomcBase

IN_COLAB = 'google.colab' in sys.modules
IN_COLAB

# Define Path Variables:
employment_data_dir = '/content/drive/My Drive/Colab Notebooks/proj2/src/data/MarketData/Employment/'
cpi_data_dir = '/content/drive/My Drive/Colab Notebooks/proj2/src/data/MarketData/CPI/'
fed_rates_dir = '/content/drive/My Drive/Colab Notebooks/proj2/src/data/MarketData/FEDRates/'
fx_rates_dir = '/content/drive/My Drive/Colab Notebooks/proj2/src/data/MarketData/FXRates/'
gdp_data_dir = '/content/drive/My Drive/Colab Notebooks/proj2/src/data/MarketData/GDP/'
ism_data_dir = '/content/drive/My Drive/Colab Notebooks/proj2/src/data/MarketData/ISM/'
sales_data_dir = '/content/drive/My Drive/Colab Notebooks/proj2/src/data/MarketData/Sales/'
treasury_data_dir = '/content/drive/My Drive/Colab Notebooks/proj2/src/data/MarketData/Treasury/'
fomc_dir = 'C:/Users/theon/GDrive/Colab Notebooks/proj2/src/data/FOMC/'
preprocessed_dir = '/content/drive/My Drive/Colab Notebooks/proj2/src/data/preprocessed/'
train_dir = '/content/drive/My Drive/Colab Notebooks/proj2/src/data/train_data/'
output_dir = '/content/drive/My Drive/Colab Notebooks/proj2/src/data/result/'
keyword_lm_dir = '/content/drive/My Drive/Colab Notebooks/proj2/src/data/LoughranMcDonald/'
glove_dir = '/content/drive/My Drive/Colab Notebooks/proj2/src/data/GloVe/'
model_dir = '/content/drive/My Drive/Colab Notebooks/proj2/src/data/models/'

class FomcTestimony(FomcBase):
    def __init__(self, verbose = True, max_threads = 20, base_dir = fomc_dir):
        super().__init__('testimony', verbose, max_threads, base_dir)

    def _get_links(self, from_year):
        self.links = []
        self.titles = []
        self.speakers = []
        self.dates = []

        if self.verbose: print("Processing request")
        to_year = datetime.today().strftime("%Y")

        if from_year < 1996:
            from_year = 1996
        elif from_year > 2006:
            print("All data from 2006 is in a single json, so return all from 2006 anyway though specified from year is ", from_year)

        url = self.base_url + '/json/ne-testimony.json'
        res = requests.get(url)
        res_list = json.loads(res.text)
        for record in res_list:
            doc_link = record.get('l')
            if doc_link:
                self.links.append(doc_link)
                self.titles.append(record.get('t'))
                self.speakers.append(record.get('s'))
                date_str = record.get('d').split(" ")[0]
                self.dates.append(datetime.strptime(date_str, '%m/%d/%Y'))

        if from_year < 2006:
            for year in range(from_year, 2006):
                url = self.base_url + '/newsevents/testimony/' + str(year) + 'testimony.htm'

                res = requests.get(url)
                soup = BeautifulSoup(res.text, 'html.parser')

                doc_links = soup.findAll('a', href=re.compile('^/boarddocs/testimony/{}/|^/boarddocs/hh/{}/'.format(str(year), str(year))))
                for doc_link in doc_links:
                    # Sometimes the same link is put for watch live video. Skip those.
                    if doc_link.find({'class': 'watchLive'}):
                        continue
                    # Add links
                    self.links.append(doc_link.attrs['href'])

                    # Handle mark-up mistakes
                    if doc_link.get('href') in ('/boarddocs/testimony/2005/20050420/default.htm'):
                        title = doc_link.get_text()
                        speaker = doc_link.parent.parent.next_element.next_element.get_text().replace('\n', '').strip()
                        date_str = doc_link.parent.parent.next_element.replace('\n', '').strip()
                    elif doc_link.get('href') in ('/boarddocs/testimony/1997/19970121.htm'):
                        title = doc_link.parent.parent.find_next('em').get_text().replace('\n', '').strip()
                        speaker = doc_link.parent.parent.find_next('strong').get_text().replace('\n', '').strip()
                        date_str = doc_link.get_text()
                    else:
                        title = doc_link.get_text()
                        speaker = doc_link.parent.find_next('div').get_text().replace('\n', '').strip()
                        # When a video icon is placed between the link and speaker
                        if speaker in ('Watch Live', 'Video'):
                            speaker = doc_link.parent.find_next('p').find_next('p').get_text().replace('\n', '').strip()
                        date_str = doc_link.parent.parent.next_element.replace('\n', '').strip()

                    self.titles.append(doc_link.get_text())
                    self.speakers.append(speaker)
                    self.dates.append(datetime.strptime(date_str, '%B %d, %Y'))

                if self.verbose: print("YEAR: {} - {} testimony docs found.".format(year, len(doc_links)))

    def _add_article(self, link, index=None):
        if self.verbose:
            sys.stdout.write(".")
            sys.stdout.flush()

        link_url = self.base_url + link
        # article_date = self._date_from_link(link)

        #print(link_url)

        # date of the article content
        # self.dates.append(article_date)

        res = requests.get(self.base_url + link)
        html = res.text
        # p tag is not properly closed in many cases
        html = html.replace('<P', '<p').replace('</P>', '</p>')
        html = html.replace('<p', '</p><p').replace('</p><p', '<p', 1)
        # remove all after appendix or references
        x = re.search(r'(<b>references|<b>appendix|<strong>references|<strong>appendix)', html.lower())
        if x:
            html = html[:x.start()]
            html += '</body></html>'
        # Parse html text by BeautifulSoup
        article = BeautifulSoup(html, 'html.parser')
        # Remove footnote
        for fn in article.find_all('a', {'name': re.compile('fn\d')}):
            # if fn.parent:
            #     fn.parent.decompose()
            # else:
            fn.decompose()
        # Get all p tag
        paragraphs = article.findAll('p')
        self.articles[index] = "\n\n[SECTION]\n\n".join([paragraph.get_text().strip() for paragraph in paragraphs])
