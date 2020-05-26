"""
AI Hub의 한국어, 영어 샘플 말뭉치 데이터셋을 활용하여, seq2seq 모델용 (ko -> en) 예제 데이터셋을 구축함.

http://aihub.or.kr/sample/KEnglish_Text_Corpus_sample.zip
"""

import os
import subprocess
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from tokenizers import ByteLevelBPETokenizer
from tokenizers.trainers import BpeTrainer

if __name__=='__main__':
    # define basic variables
    data_dir = Path('data')
    download_shell = f"wget http://aihub.or.kr/sample/KEnglish_Text_Corpus_sample.zip -P {data_dir} && cd {data_dir} && unzip KEnglish_Text_Corpus_sample.zip"
    
    # check the data directory
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    
    # download data
    subprocess.check_output([download_shell], shell=True)
    
    # load data and train, dev split
    df = pd.DataFrame([], columns=['ko', 'en'])
    mapper = {'한국어':'ko', '영어검수':'en','영어 검수':'en', '원문':'ko', 'REVIEW':'en', 'Review':'en'}
    for fn in Path('data').glob('*.xlsx'):
        cols = pd.read_excel(fn, nrows=0).columns.intersection(mapper.keys()).tolist()
        sub_df = pd.read_excel(fn, usecols=cols).rename(columns=mapper)
        df = pd.concat((df, sub_df))
        
    train_df, dev_df = train_test_split(df, test_size=0.1,random_state=42)
    train_df.to_csv(data_dir/'train.tsv', sep='\t', index=False)
    dev_df.to_csv(data_dir/'dev.tsv', sep='\t', index=False)
    
    #### train tokenizer    
    # ko
    ko_path = data_dir / 'ko'
    if not os.path.exists(ko_path):
        os.mkdir(ko_path)
    with open(ko_path / 'tokenizer_train_ko.txt', 'w', encoding='utf-8') as f:
        for line in df['ko'].tolist():
            print(line, file=f)
    
    tokenizer = ByteLevelBPETokenizer(dropout=0.1,  # dropout bpe
                                      unicode_normalizer='nfkc') # nfkc
    
    tokenizer.train(
            files=[ko_path / 'tokenizer_train_ko.txt'],
            vocab_size=10000,
            min_frequency=2,
            show_progress=True,
            special_tokens = ['[PAD]', '[SOS]', '[EOS]', '[UNK]']
                    )
    tokenizer.model.save(ko_path)
    
    # en
    en_path = data_dir / 'en'
    if not os.path.exists(en_path):
        os.mkdir(en_path)
    with open(en_path / 'tokenizer_train_en.txt', 'w', encoding='utf-8') as f:
        for line in df['en'].tolist():
            print(line, file=f)

    tokenizer = ByteLevelBPETokenizer(dropout=0.1,  # dropout bpe
                                      unicode_normalizer='nfkc') # nfkc
        tokenizer.train(
            files=[ko_path / 'tokenizer_train_en.txt'],
            vocab_size=10000,
            min_frequency=2,
            show_progress=True,
            special_tokens = ['[PAD]', '[SOS]', '[EOS]', '[UNK]']
                    )
    tokenizer.model.save(ko_path)

    