from fastai import *
from fastai.text import *
from pathlib import Path
import pandas as pd
import numpy as np
import re


df = pd.read_csv('commbackup_v1.csv')

# Data Processing
def convertrating(row):
  if row['rating']>=5:
    return pd.Series(['FAVOR', ''.join(i for i in row['review'] if ord(i) < 128)])
  else:
    return pd.Series(['AGAINST', ''.join(i for i in row['review'] if ord(i) < 128)])
df['rating'] = df.apply(convertrating, axis=1)
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]
train[['review', 'rating']].to_csv('train_comments.csv')
test[['review', 'rating']].to_csv('test_comments.csv')
import torch
print("Cuda available" if torch.cuda.is_available() is True else "CPU")
print("PyTorch version: ", torch.__version__)

train[['rating', 'review']].to_csv('train.csv', index=False, header=False)


data_lm = TextLMDataBunch.from_csv('', 'train.csv', min_freq=1)
data_lm.save()
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)
learn.fit_one_cycle(cyc_len=1, max_lr=1e-3, moms=(0.8, 0.7))
learn.unfreeze()
learn.fit_one_cycle(cyc_len=20, max_lr=1e-3, moms=(0.8, 0.7))
learn.save_encoder('ft_enc')
data_clas = TextClasDataBunch.from_csv('', 'train.csv', vocab=data_lm.train_ds.vocab, min_freq=1, bs=32)
data_clas.save()
learn = text_classifier_learner(data_clas, drop_mult=0.5, arch=AWD_LSTM)
learn.load_encoder('ft_enc')
learn.freeze()
learn.fit_one_cycle(cyc_len=1, max_lr=1e-3, moms=(0.8, 0.7))
learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(1e-4,1e-2), moms=(0.8,0.7))
learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(1e-5,5e-3), moms=(0.8,0.7))
learn.unfreeze()
learn.fit_one_cycle(4, slice(1e-5,1e-3), moms=(0.8,0.7))

def get_predictions(row):
  stance, cat, probs = learn.predict(row['review'])
  return pd.Series([stance, probs[0].item(), probs[1].item()])
tmp[['rating', 'probfavor', 'probagainst']] = tmp.apply(get_predictions, axis=1)
# tmp['rating'] = tmp['review'].apply(lambda row: str(learn.predict(row)[0]))	

tmp['rating_original'] = test['rating']
test_merged = pd.merge(test, df[['paperid', 'rating']], on=['paperid'])
tmp['rating_value'] = test_merged['rating_y']

tmp.to_csv('results.csv')