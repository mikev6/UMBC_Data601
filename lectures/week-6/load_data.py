import requests # for loading data from an online resource
from io import StringIO # for reading inputs
import pandas as pd # for manipulating data

def data(url = "https://drive.google.com/file/d/1XdOaGHuGWgEMVMEiJ0dbwdPh6mH6PQZi/view?usp=sharing"):

    orig_url = url

    file_id = orig_url.split('/')[-2]

    dwn_url='https://drive.google.com/uc?export=download&id=' + file_id

    url = requests.get(dwn_url).text

    csv_raw = StringIO(url)

    advertising = pd.read_csv(csv_raw, index_col= 0)

    return advertising