import pandas as pd
import numpy as np
from argparse import ArgumentParser
from utils import bag_of_words, split_doc, clean, get_voc
from model import BernoulliNB

parser = ArgumentParser()
parser.add_argument('-t','--text', help='Enter your text.', default="patato is sweet", required=True, type=str)
main_args = vars(parser.parse_args())

# print(type(main_args))

text= main_args["text"]
print(text)

# print(text)

corpus= pd.read_csv("./data/corpus.txt", header=0)
corpus_clean= clean(corpus)

# print(corpus_clean)
splitted_corp= split_doc(corpus_clean)
voc= get_voc(splitted_corp)
X= np.array([bag_of_words(split_doc(corpus_clean)[i], voc) for i in range(len(split_doc(corpus_clean)))])
y= corpus_clean[["Class"]].values


nb = BernoulliNB(voc=voc,alpha=0.0).fit(np.where(X > 0, 1, 0), y) # Binarizing X

# print(nb.feature_prob_)
## make prediction
print(nb.predic_new(text))





