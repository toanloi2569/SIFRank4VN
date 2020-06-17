#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge"
# Date: 2019/6/19
import nltk
from model import input_representation

GRAMMAR1 = """  NP:
        {<V|N.*>{1,3}<PC><V|N.*>{1,2}}
        
        {<N.*>{1,2}<M>}
        {<M><N.*>{1,2}}
        {<M>}
        
        {<P><N.*>{0,2}}
        
        {<V|N.*>{1,6}}
        """
GRAMMAR2 = """  NP:
        {<V><N.*>{1, 4}}
        """

def is_subset(start, end, keyphrase_candidate):
    for _, (s, e) in keyphrase_candidate:
        if s <= start <= e or s <= end <= e:
            return True
    return False

def extract_candidates(tokens_tagged, no_subset=True):
    """
    Based on part of speech return a list of candidate phrases
    :param text_obj: Input text Representation see @InputTextObj
    :param no_subset: if true won't put a candidate which is the subset of an other candidate
    :return keyphrase_candidate: list of list of candidate phrases: [tuple(string,tuple(start_index,end_index))]
    """

    keyphrase_candidate = set()
    list_np_pos_tag_tokens = []
    
    # nhóm các cụm theo mẫu GAMMAR1 lại và gắn cho nó nhãn NP
    np_parser1 = nltk.RegexpParser(GRAMMAR1)  # Dùng hàm RegexpParser của nltk để phân cụm các cụm danh từ (NP)
    list_np_pos_tag_tokens.append(np_parser1.parse(tokens_tagged))
    
    np_parser2 = nltk.RegexpParser(GRAMMAR2)
    list_np_pos_tag_tokens.append(np_parser2.parse(tokens_tagged))
    
    for np_pos_tag_tokens in list_np_pos_tag_tokens:
        count = 0
        for token in np_pos_tag_tokens:
            if (isinstance(token, nltk.tree.Tree) and token._label == "NP"):
                np = ' '.join(word for word, tag in token.leaves())
                length = len(token.leaves())
                start_end = (count, count + length)
                count += length
                if no_subset and is_subset(count, count + length, keyphrase_candidate):
                    continue
                keyphrase_candidate.add((np, start_end))

            else:
                count += 1
    return list(keyphrase_candidate)

