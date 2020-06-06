#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge_sy"
# Date: 2020/2/21


import sys
sys.path.append('.')

import time
import os

from vncorenlp import VnCoreNLP

from embeddings import sent_emb_sif, word_emb_phoBert
from model.method import SIFRank, SIFRank_plus


path = os.path.dirname(os.path.realpath('__file__'))
vncorenlp = VnCoreNLP(path+"/auxiliary_data/VnCoreNLP-master/VnCoreNLP-1.1.1.jar", annotators="wseg, pos", max_heap_size='-Xmx500m') 

phoBERT = word_emb_phoBert.WordEmbeddings()
SIF = sent_emb_sif.SentEmbeddings(phoBERT, lamda=1.0, embeddings_type='bert')
try:
    text = '''
    Cụ_thể , theo Tổng_cục Hải_quan , để thực_hiện ý_kiến chỉ_đạo của Ban cán_sự Đảng_Bộ Tài_chính tại cuộc họp ngày 26/5/2020 , của Bộ_trưởng Bộ Tài_chính tại công_văn số 6170 / BTC - TCT về công_tác cán_bộ tại Cục Hải_quan tỉnh Bắc_Ninh , Tổng_cục Hải_quan đã có các quyết_định tạm đình_chỉ công_tác 15 ngày ( kể từ ngày 27/5/2020 ) đối_với các trường_hợp sau.Tạm đình_chỉ chức_vụ đối_với ông Trần_Thành_Tô , Cục trưởng Cục Hải_quan tỉnh Bắc_Ninh , người ký quyết_định thành_lập Đoàn kiểm_tra sau thông_quan Công_ty Tenma để tập_trung tổ_chức phục_vụ đoàn Thanh_tra Bộ Tài_chính và rà_soát , kiểm_điểm , làm rõ trách_nhiệm của đoàn này và người đứng đầu liên_quan trong việc kiểm_soát sau thông_quan đối_với Công_ty TNHH Tenma_Việt Nam. Các cá_nhân khác là công_chức của Cục Hải_quan Bắc_Ninh gồm ông Dương_Minh_Khải , Đội_trưởng Đội Kiểm_soát Hải_quan , ông Nguyễn_Văn_Phúc - Phó trưởng phòng Nghiệp_vụ , ông Vũ_Quang Hà , đội_trưởng đội nghiệp_vụ , Chi_cục Hải_quan Bắc_Ninh , ông Nguyễn_Lưu_Bình_Trọng , công_chức Chi_cục Hải_quan cảng nội_địa Tiên_Sơn , bà Nguyễn_Thị_Hảo , công_chức Văn_phòng Cục Hải_quan tỉnh Bắc_Ninh ... Tất_cả các cán_bộ , công_chức trên bị tạm đình_chỉ chức_vụ để kiểm_điểm , làm rõ trách_nhiệm trong việc kiểm_tra sau thông_quan đối_với Công_ty TNHH Tenma_Việt Nam.Cũng trong chiều ngày 26/5 , Bộ Tài_chính chỉ_đạo Tổng_cục Thuế tạm đình_chỉ 15 ngày đối_với lãnh_đạo Cục Thuế_Bắc_Ninh , người ký quyết_định kiểm_tra Công_ty Tenma Việt_Nam , khơi_mào nghi_vấn trốn_thuế của doanh_nghiệp này.Theo đó , Bộ_trưởng Bộ Tài_chính Đinh_Tiến_Dũng yêu_cầu Tổng_cục trưởng Tổng_cục Thuế tạm đình_chỉ công_tác 15 ngày đối_với các công_chức tham_gia đoàn đoàn kiểm_tra thuế.Đồng thời , lãnh_đạo Bộ yêu_cầu Tổng_cục Thuế tạm đình_chỉ lãnh_đạo Cục Thuế tỉnh Bắc_Ninh ký quyết_định thành_lập đoàn kiểm_tra thuế tại Công_ty Tenma Việt_Nam để thực_hiện công_tác kiểm_điểm , phục_vụ công_tác thanh , kiểm_tra theo quy_định khi sự_việc đang được xử lý.Bộ Tài_chính yêu_cầu trong trường_hợp công_chức bị đình_chỉ công_tác là thủ_trưởng đơn_vị , đề_nghị Tổng_cục trưởng , Tổng_cục Thuế lựa_chọn 1 lãnh_đạo để gia phụ_trách điều_hành đơn vị.Như vậy , theo chỉ_đạo của Bộ_trưởng Bộ Tài_chính , Tổng_cục Hải_quan đã tạm đình_chỉ chức_vụ đối_với Cục trưởng Cục Hải_quan Bắc Ninh.Hiện vẫn chưa rõ danh_tính vị lãnh_đạo Cục Thuế_Bắc_Ninh bị lãnh_đạo Bộ Tài_chính chỉ_đạo đình_chỉ chức_vụ . Theo nguồn tin từ Tổng_cục Thuế , ngày_mai 27/5 , Tổng_cục Thuế sẽ có thông_tin chính_thức về các lãnh_đạo và cán_bộ của Cục Thuế_Bắc_Ninh bị đình_chỉ chức_vụ theo đúng tinh_thần chỉ_đạo của Bộ Tài chính.Báo Dân_trí sẽ thông_tin kịp_thời đến độc_giả những diễn_biến mới nhất của vụ_việc .
    '''
    keyphrases = SIFRank_plus(text.replace('_',' ').replace('.','. '), SIF, vncorenlp, N=15)
    # keyphrases_ = SIFRank_plus(text, SIF, en_model, N=15, elmo_layers_weight=elmo_layers_weight)
except Exception as e:
    print(e.message)

vncorenlp.close()
print(keyphrases)
# print(keyphrases_)