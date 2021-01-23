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

class FomcStatement(FomcBase):
    def __init__(self, verbose = True, max_threads = 20, base_dir = fomc_dir):
        super().__init__('statement', verbose, max_threads, base_dir)

    def _get_links(self, from_year):
        self.links = []
        self.titles = []
        self.speakers = []
        self.dates = []

        r = requests.get(self.calendar_url)
        soup = BeautifulSoup(r.text, 'html.parser')
        if self.verbose: print("Getting links for statements...")
        contents = soup.find_all('a', href=re.compile('^/newsevents/pressreleases/monetary\d{8}[ax].htm'))
        print(contents)
        self.links = [content.attrs['href'] for content in contents]
        self.speakers = [self._speaker_from_date(self._date_from_link(x)) for x in self.links]
        self.titles = ['FOMC Statement'] * len(self.links)
        self.dates = [datetime.strptime(self._date_from_link(x), '%Y-%m-%d') for x in self.links]
        for i, m_date in enumerate(self.dates):
            if m_date == datetime(2019,10,11):
                self.dates[i] = datetime(2019,10,4)

        if self.verbose: print("{} links found in the current page.".format(len(self.links)))

        if from_year <= 2014:
            for year in range(from_year, 2020):
                yearly_contents = []
                fomc_yearly_url = self.base_url + '/monetarypolicy/fomchistorical' + str(year) + '.htm'
                r_year = requests.get(fomc_yearly_url)
                soup_yearly = BeautifulSoup(r_year.text, 'html.parser')
                yearly_contents = soup_yearly.findAll('a', text = 'Statement')
                for yearly_content in yearly_contents:
                    self.links.append(yearly_content.attrs['href'])
                    self.speakers.append(self._speaker_from_date(self._date_from_link(yearly_content.attrs['href'])))
                    self.titles.append('FOMC Statement')
                    self.dates.append(datetime.strptime(self._date_from_link(yearly_content.attrs['href']), '%Y-%m-%d'))
                    if self.dates[-1] == datetime(2007,6,18):
                        self.dates[-1] = datetime(2007,6,28)
                    elif self.dates[-1] == datetime(2007,8,17):
                        self.dates[-1] = datetime(2007,8,16)
                    elif self.dates[-1] == datetime(2008,1,22):
                        self.dates[-1] = datetime(2008,1,21)
                    elif self.dates[-1] == datetime(2008,3,11):
                        self.dates[-1] = datetime(2008,3,10)
                    elif self.dates[-1] == datetime(2008,10,8):
                        self.dates[-1] = datetime(2008,10,7)

                if self.verbose: print("YEAR: {} - {} links found.".format(year, len(yearly_contents)))

        print("There are total ", len(self.links), ' links for ', self.content_type)

    def _add_article(self, link, index=None):
        if self.verbose:
            sys.stdout.write(".")
            sys.stdout.flush()

        res = requests.get(self.base_url + link)
        html = res.text
        article = BeautifulSoup(html, 'html.parser')
        paragraphs = article.findAll('p')
        self.articles[index] = "\n\n[SECTION]\n\n".join([paragraph.get_text().strip() for paragraph in paragraphs])
