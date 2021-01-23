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

class FomcMeetingScript(FomcBase):
    def __init__(self, verbose = True, max_threads = 20, base_dir = fomc_dir):
        super().__init__('meeting_script', verbose, max_threads, base_dir)

    def _get_links(self, from_year):
        self.links = []
        self.titles = []
        self.speakers = []
        self.dates = []

        r = requests.get(self.calendar_url)
        soup = BeautifulSoup(r.text, 'html.parser')

        # Meeting Script can be found only in the archive as it is published after five years
        if from_year > 2014:
            print("Meeting scripts are available for 2014 or older")
        if from_year <= 2014:
            for year in range(from_year, 2015):
                yearly_contents = []
                fomc_yearly_url = self.base_url + '/monetarypolicy/fomchistorical' + str(year) + '.htm'
                r_year = requests.get(fomc_yearly_url)
                soup_yearly = BeautifulSoup(r_year.text, 'html.parser')
                meeting_scripts = soup_yearly.find_all('a', href=re.compile('^/monetarypolicy/files/FOMC\d{8}meeting.pdf'))
                for meeting_script in meeting_scripts:
                    self.links.append(meeting_script.attrs['href'])
                    self.speakers.append(self._speaker_from_date(self._date_from_link(meeting_script.attrs['href'])))
                    self.titles.append('FOMC Meeting Transcript')
                    self.dates.append(datetime.strptime(self._date_from_link(meeting_script.attrs['href']), '%Y-%m-%d'))
                if self.verbose: print("YEAR: {} - {} meeting scripts found.".format(year, len(meeting_scripts)))
            print("There are total ", len(self.links), ' links for ', self.content_type)

    def _add_article(self, link, index=None):
        if self.verbose:
            sys.stdout.write(".")
            sys.stdout.flush()

        link_url = self.base_url + link
        pdf_filepath = self.base_dir + 'script_pdf/FOMC_MeetingScript_' + self._date_from_link(link) + '.pdf'

        # Scripts are provided only in pdf. Save the pdf and pass the content
        res = requests.get(link_url)
        with open(pdf_filepath, 'wb') as f:
            f.write(res.content)

        # Extract text from the pdf
        # pdf_file_parsed = parser.from_file(pdf_filepath)
        # paragraphs = re.sub('(\n)(\n)+', '\n', pdf_file_parsed['content'].strip())
        pdf_file_parsed = textract.process(pdf_filepath).decode('utf-8')
        paragraphs = re.sub('(\n)(\n)+', '\n', pdf_file_parsed.strip())
        paragraphs = paragraphs.split('\n')

        section = -1
        paragraph_sections = []
        for paragraph in paragraphs:
            if not re.search('^(page|january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)', paragraph.lower()):
                if len(re.findall(r'[A-Z]', paragraph[:10])) > 5 and not re.search('(present|frb/us|abs cdo|libor|rp–ioer|lsaps|cusip|nairu|s cpi|clos, r)', paragraph[:10].lower()):
                    section += 1
                    paragraph_sections.append("")
                if section >= 0:
                    paragraph_sections[section] += paragraph
        self.articles[index] = "\n\n[SECTION]\n\n".join([paragraph for paragraph in paragraph_sections])
