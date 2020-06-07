#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge_sy"
# Date: 2020/2/21


import sys
# sys.path.append('.')
import time
from vncorenlp import VnCoreNLP
from embeddings import sent_emb_sif, word_emb_phoBert
from model.method import SIFRank, SIFRank_plus

class SIFRank4VN():
    def __init__(self):
        # path = os.path.dirname(os.path.realpath('__file__'))
        self.vncorenlp = VnCoreNLP("auxiliary_data/VnCoreNLP-master/VnCoreNLP-1.1.1.jar", annotators="wseg, pos", max_heap_size='-Xmx500m') 
        self.phoBERT = word_emb_phoBert.WordEmbeddings()
        self.SIF = sent_emb_sif.SentEmbeddings(self.phoBERT, lamda=1.0, embeddings_type='bert')


    def sifrank_extract(self, text, nphrase=15, ratio=0.6):
        keyphrases = SIFRank(text, self.SIF, self.vncorenlp, N=nphrase, ratio=ratio)
        return keyphrases

    def sifrank_plus_extract(self, text, nphrase=15, ratio=0.6):
        keyphrases = SIFRank_plus(text, self.SIF, self.vncorenlp, N=nphrase, ratio=ratio)
        return keyphrases

    def close_vncorenlp(self):
        self.vncorenlp.close()
