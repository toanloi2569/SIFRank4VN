# SIFRank4VN

Edit from https://github.com/sunyilgdx/SIFRank to use for Vietnamese  
Keyphrase extraction for Vietnamese

## Requirements and Installation

- Python version >= 3.6
- Java 1.8+
  
```[python]
pip install torch torchvision
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
pip install fastBPE
pip install vncorenlp
pip install python-Levenshtein
```

Clone SIFRank4VN  
Clone [vncorenlp](https://github.com/vncorenlp/VnCoreNLP.git) and copy to folder *auxiliary_data/*  
Download model [PhoBERT base fairseq](https://github.com/VinAIResearch/PhoBERT#using-phobert-in-fairseq-) and copy to folder *auxiliary_data/*  

## Usage

```[python]
import os
os.chdir('./SIFRank4VN')

from SIFRank4VN.test.test import SIFRank4VN

text = '''
Theo số liệu thống kê của Đại học Johns Hopkins, trong tháng 4, chưa bao giờ số ca mắc mới Covid-19 vượt 100.000 ca/ngày, tuy nhiên kể từ ngày 21/5, chỉ có 5 ngày con số này dưới mốc 100.000 ca. Số ca mắc mới Covid-19 toàn cầu trong ngày đạt kỷ lục 130.400 ca vào hôm 3/6.

Số ca mắc mới tăng mạnh một phần được cho là do các nước tăng năng lực xét nghiệm, nhưng tại nhiều quốc gia, năng lực xét nghiệm còn hạn chế và vẫn chưa phản ánh đúng quy mô của đại dịch Covid-19.

Số ca mắc mới Covid-19 tại nhiều quốc gia từng là tâm dịch như Trung Quốc, Mỹ, Anh, Italia, Tây Ban Nha, Pháp, bắt đầu tăng chậm lại. Trong khi đó, tại nhiều nước, đặc biệt ở khu vực Nam Mỹ, Trung Đông và châu Phi, dịch tiếp tục bùng phát mạnh.
'''

sifrank_extractor = SIFRank4VN()
phrases = sifrank_extractor.sifrank_plus_extract(text, ratio=0.6, nphrase=10)

print (phrases)
sifrank_extractor.close_vncorenlp()
```
