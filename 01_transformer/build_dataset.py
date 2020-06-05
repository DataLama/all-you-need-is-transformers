"""
AI Hub의 한국어, 영어 샘플 말뭉치 데이터셋을 활용하여, seq2seq 모델용 (ko -> en) 예제 데이터셋을 구축함.

http://aihub.or.kr/sample/KEnglish_Text_Corpus_sample.zip
"""

import os
import sys
import subprocess
import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format = '[%(asctime)s][%(levelname)s] %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)

if __name__=='__main__':
    # dataset
    logging.info("Define Basic Variables")
    data_dir = Path('data')
    download_shell = f"wget http://aihub.or.kr/sample/KEnglish_Text_Corpus_sample.zip -P {data_dir} && cd {data_dir} && unzip KEnglish_Text_Corpus_sample.zip"
    
    if not os.path.exists(data_dir):
        logging.info("make new directory.")
        os.mkdir(data_dir)
    
    if not os.path.exists(data_dir / 'KEnglish_Text_Corpus_sample.zip'):
        logging.info("download the data.")
        subprocess.check_output([download_shell], shell=True)

    logging.info("load data and split the train and dev set.")
    df = pd.DataFrame([], columns=['ko', 'en'])
    mapper = {'한국어':'ko', '영어검수':'en','영어 검수':'en', '원문':'ko', 'REVIEW':'en', 'Review':'en'}
    for fn in Path('data').glob('*.xlsx'):
        cols = pd.read_excel(fn, nrows=0).columns.intersection(mapper.keys()).tolist()
        sub_df = pd.read_excel(fn, usecols=cols).rename(columns=mapper)
        df = pd.concat((df, sub_df))
    
    train_df, dev_df = train_test_split(df, test_size=0.1,random_state=42)
    train_df.to_csv(data_dir/'train.tsv', sep='\t', index=False)
    dev_df.to_csv(data_dir/'dev.tsv', sep='\t', index=False)
  