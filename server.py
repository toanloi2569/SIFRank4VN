import sys
from flask import Flask, jsonify, request, render_template
from werkzeug.utils import secure_filename
import json  
import os

from vncorenlp import VnCoreNLP
from embeddings import sent_emb_sif, word_emb_phoBert
from model.method import SIFRank, SIFRank_plus

app = Flask(__name__, static_url_path="", static_folder='./static/')
vncorenlp = VnCoreNLP(
    "auxiliary_data/VnCoreNLP-master/VnCoreNLP-1.1.1.jar", 
    annotators="wseg, pos", 
    max_heap_size='-Xmx500m'
) 
phoBERT = word_emb_phoBert.WordEmbeddings()
SIF = sent_emb_sif.SentEmbeddings(phoBERT, lamda=1.0, embeddings_type='bert')

# Xử lý điều hướng câu truy vấn từ client gửi đến
@app.route('/extract', methods=['GET'])
def extract():
    return render_template('extract.html')

@app.route('/result', methods=['POST'])
def result():
    text = request.form.get('query')
    highlighted, phrases = extract_keyphrase(text)
    return render_template('result.html', highlighted=highlighted, phrases=phrases)

def extract_keyphrase(text):
    phrases = []
    highlighted = text
    try:
        keyphrases = SIFRank_plus(text, SIF, vncorenlp, position_bias=3.4, N=10, ratio=0.6)
        for keyphrase in keyphrases:
            phrases.append({'phrase':keyphrase[0], 'score': round(keyphrase[1], 3)})
            pos = highlighted.lower().find(keyphrase[0].replace('_', ' '))
            highlighted = add_mark_tag(highlighted, pos, len(keyphrase[0]))
    except:
        pass
    
    return highlighted, phrases

def add_mark_tag(text, pos, length):
    return text[:pos]+'<mark>'+text[pos:(pos+length)]+'</mark>'+text[(pos+length):] if pos!=-1 else text

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=5000, type=int, help='port listening')
    args = parser.parse_args()
    port = args.port
    app.debug = False
    app.run(host='0.0.0.0', port=port)
    
