import numpy as np
import pandas as pd
import collections

def clean(corpus):
  '''
  split the text and the labels

  '''
  x= np.array([corpus.iloc[i][0].split('.')[0] for i in range(corpus.shape[0])]).T
  y= np.array([int(corpus.iloc[i][0].split('.')[1].replace(' ','')) for i in range(corpus.shape[0])]).T
  corpus['Text']= x
  corpus['Class']= y
  return corpus


def split_doc(corpus):
  # Cleaned corpus
  splitted_docs= [corpus.iloc[i][0].split(" ") for i in range(corpus.shape[0])]

  for i in splitted_docs:
    for k in range(len(i)):
      i[k]= i[k].lower()
  return splitted_docs


def get_voc(splitted_docs):
  #splitted_docs
  voc= set()
  for docs in splitted_docs:
    for word in docs:
        voc.add(word.lower())
  voc = list(voc)
  return voc

def bag_of_words(sent, voc):
    '''
    check which words from the voc appear in the sent

    '''
    # corpus_clean= 
    # splitted_corp= split_doc(corpus_clean)
    # voc= get_voc(splitted_corp)

    index_word = {}
    i = 0
    for word in voc:
        index_word[word] = i 
        i += 1
  
    count_dict = collections.defaultdict(int)
    vec = np.zeros(len(voc))
    for item in sent:
      if count_dict[item] < 1:
        count_dict[item] += 1
    for key,item in count_dict.items():
        vec[index_word[key]] = item
    return list(vec)

