#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge"
# Date: 2019/6/19

from model import extractor

with open("./auxiliary_data/stopwords_vietnamese.txt", 'r', encoding='UTF-8') as f:
    data = f.read()
    stopwords = data.split('\n')
    
class InputTextObj:
    """Represent the input text in which we want to extract keyphrases"""

    def __init__(self, preprocess_model, text=""):
        """
        :param is_sectioned: If we want to section the text.
        :param en_model: the pipeline of tokenization and POS-tagger
        :param considered_tags: The POSs we want to keep
        """
        punctuation_connect = ["&","và"]
        self.tokens = []
        self.tokens_tagged = []

        pos = preprocess_model.pos_tag(text)
        self.tokens = [_[0] for sent in pos for _ in sent]
        self.tokens_tagged = [(_[0], _[1]) for sent in pos for _ in sent]
        assert len(self.tokens) == len(self.tokens_tagged)

        for i, token in enumerate(self.tokens):
            if token.lower() in stopwords:
                self.tokens_tagged[i] = (token, "IN")
            if token.lower() in punctuation_connect:
                self.tokens_tagged[i] = (token,"PC")
        ### Trích xuất các cụm danh từ (NP)
        self.keyphrase_candidate = extractor.extract_candidates(self.tokens_tagged)

