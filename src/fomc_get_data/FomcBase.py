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

class FomcBase(metaclass=ABCMeta):
    def __init__(self, content_type, verbose, max_threads, base_dir = fomc_dir):

        # Set arguments to internal variables
        self.content_type = content_type
        self.verbose = verbose
        self.MAX_THREADS = max_threads
        self.base_dir = base_dir

        # Initialization
        self.df = None
        self.links = None
        self.dates = None
        self.articles = None
        self.speakers = None
        self.titles = None

        # FOMC website URLs
        self.base_url = 'https://www.federalreserve.gov'
        self.calendar_url = self.base_url + '/monetarypolicy/fomccalendars.htm'

        # FOMC Chairperson's list
        self.chair = pd.DataFrame(
            data=[["Greenspan", "Alan", "1987-08-11", "2006-01-31"],
                  ["Bernanke", "Ben", "2006-02-01", "2014-01-31"],
                  ["Yellen", "Janet", "2014-02-03", "2018-02-03"],
                  ["Powell", "Jerome", "2018-02-05", "2022-02-05"]],
            columns=["Surname", "FirstName", "FromDate", "ToDate"])

    def _date_from_link(self, link):
        date = re.findall('[0-9]{8}', link)[0]
        if date[4] == '0':
            date = "{}-{}-{}".format(date[:4], date[5:6], date[6:])
        else:
            date = "{}-{}-{}".format(date[:4], date[4:6], date[6:])
        return date

    def _speaker_from_date(self, article_date):
        if self.chair.FromDate[0] < article_date and article_date < self.chair.ToDate[0]:
            speaker = self.chair.FirstName[0] + " " + self.chair.Surname[0]
        elif self.chair.FromDate[1] < article_date and article_date < self.chair.ToDate[1]:
            speaker = self.chair.FirstName[1] + " " + self.chair.Surname[1]
        elif self.chair.FromDate[2] < article_date and article_date < self.chair.ToDate[2]:
            speaker = self.chair.FirstName[2] + " " + self.chair.Surname[2]
        elif self.chair.FromDate[3] < article_date and article_date < self.chair.ToDate[3]:
            speaker = self.chair.FirstName[3] + " " + self.chair.Surname[3]
        else:
            speaker = "other"
        return speaker

    @abstractmethod
    def _get_links(self, from_year):
        pass

    @abstractmethod
    def _add_article(self, link, index=None):
        pass

    def _get_articles_multi_threaded(self):
        if self.verbose:
            print("Processing request----------")

        self.articles = ['']*len(self.links)
        jobs = []
        index = 0
        while index < len(self.links):
            if len(jobs) < self.MAX_THREADS:
                t = threading.Thread(target=self._add_article, args=(self.links[index],index,))
                jobs.append(t)
                t.start()
                index += 1
            else:
                t = jobs.pop(0)
                t.join()
        for t in jobs:
            t.join()

        #for row in range(len(self.articles)):
        #    self.articles[row] = self.articles[row].strip()

    def get_contents(self, from_year=1990):
        self._get_links(from_year)
        self._get_articles_multi_threaded()
        dict = {
            'date': self.dates,
            'contents': self.articles,
            'speaker': self.speakers,
            'title': self.titles
        }
        self.df = pd.DataFrame(dict).sort_values(by=['date'])
        self.df.reset_index(drop=True, inplace=True)
        return self.df

    def pickle_dump_df(self, filename="output.pickle"):
        filepath = self.base_dir + filename
        print("")
        if self.verbose: print("Writing to ", filepath)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as output_file:
            pickle.dump(self.df, output_file)

    def save_texts(self, prefix="FOMC_", target="contents"):
        tmp_dates = []
        tmp_seq = 1
        for i, row in self.df.iterrows():
            cur_date = row['date'].strftime('%Y-%m-%d')
            if cur_date in tmp_dates:
                tmp_seq += 1
                filepath = self.base_dir + prefix + cur_date + "-" + str(tmp_seq) + ".txt"
            else:
                tmp_seq = 1
                filepath = self.base_dir + prefix + cur_date + ".txt"
            tmp_dates.append(cur_date)
            if self.verbose: print("Writing to ", filepath)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w", encoding='utf-8') as output_file:
                output_file.write(row[target])
