# -*- coding: utf-8 -*-
#Import packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup
import urllib2
from urllib2 import HTTPError
import re
import timeit
import time
import requests
urls= ["https://www.researchgate.net/profile/Virach_Sornlertlamvanich/publications/"]
ROOT_URL = 'https://www.researchgate.net/'


def rg2paperList(url):
    papers = ['go','yeah']
    i=1
    paper_list = []
    while(len(papers)>0 and len(paper_list)<5):
        html_doc = urllib2.urlopen(url+str(i)).read()
        soup = BeautifulSoup(html_doc, 'html.parser')
        papers = soup.find_all('a',text=re.compile('Download full-text'))
        for paper in papers:
            paper_list.append(paper['href'])
        i+=1
    return paper_list

start = timeit.default_timer()
papers_url = rg2paperList(urls[0])
stop = timeit.default_timer()
print("Execution time : "+str(round(stop - start,2))+"s.")

def paperList2pdfList(papers_url):
    pdf_list = []
    for paper in papers_url[0:5]:
        html_doc = urllib2.urlopen(ROOT_URL+paper).read()
        soup = BeautifulSoup(html_doc, 'html.parser')
        pdfs = soup.find_all('a',text=re.compile('Download full-text PDF'))
        for pdf in pdfs:
            pdf_list.append(pdf['href'])
                       
start = timeit.default_timer()
pdfs_url = paperList2pdfList(papers_url)
stop = timeit.default_timer()
print("Execution time : "+str(round(stop - start,2))+"s.")