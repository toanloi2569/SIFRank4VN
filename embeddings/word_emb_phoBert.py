from fairseq.models.roberta import RobertaModel
from fairseq.data.encoders.fastbpe import fastBPE  
from fairseq import options  
import numpy as np

class WordEmbeddings():
    def __init__(self, pretrain = "auxiliary_data/PhoBERT_base_fairseq"):
        self.phoBERT = RobertaModel.from_pretrained(pretrain, checkpoint_file='model.pt')
        parser = options.get_preprocessing_parser()  
        parser.add_argument('--bpe-codes', type=str, help='path to fastBPE BPE', default=pretrain+"/bpe.codes")  
        args, unknown = parser.parse_known_args()  
        self.phoBERT.bpe = fastBPE(args) #Incorporate the BPE encoder into PhoBERT

    def get_tokenized_words_embeddings(self, sents_tokened):
        # dimension of word is [len(sents), 1, max(len(words)), sizeofvector]
        emb = []
        max_sent = max([len(sent) for sent in sents_tokened])
        zeros = [0 for i in range(768)]

        for sent_tokened in sents_tokened:
            text = ' '.join(sent_tokened)
            words = self.phoBERT.extract_features_aligned_to_words(text)  
            words_embedding = []
            
            if len(words) != len(sent_tokened):
                pos = 0
                for token in sent_tokened:
                    while pos < len(words) and token.find(words[pos].text) == -1:
                        pos += 1
                    words_embedding.append(words[pos].vector.tolist())
                    pos += 1
            else:
                words_embedding = [word.vector.tolist() for word in words]
            
            for i in range(max_sent - len(words_embedding)):
                words_embedding.append(zeros)

            emb.append([words_embedding])
        
        return np.array(emb), (len(emb), max_sent)