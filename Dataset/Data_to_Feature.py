from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import pipeline
import torch
import random
import time
import pandas as pd
from matplotlib import pyplot as plt
import json
import numpy as np
import nltk
from nltk import sent_tokenize
import time

#Tokenizer 사용 설정
nltk.download("punkt")

#Get_csv
essay_data_file = "BenchMark/Bench_mark_IELTS.csv"
df = pd.read_csv(filepath_or_buffer=essay_data_file)
#모든 행 보기
pd.set_option('display.max_columns', None)

print(df)

#print(Review.iloc[2].split("\"")[1])

#각 종류별 15개의 데이터가 있음 i = 14

#coulmns info
#id : 식별자, text : 학생 (사람)이 쓴 에세이 추출)
#Instruction = prompt, #source_text = AIGT


#총 3*2421개의 데이터를 만들 것 이므로 1텍스트당 3개의 문장을 뽑는다.
#(각 문장이 48토큰이 안 되서 좀 많이해도 됨)


#Huggin Face 에서 지원하는 Pre-trained Model의 라이브러리인 Transformer
#Pipeline (파이프라인)을 이용해 pretranied task를 로딩한다.

AI_snt = []
Human_snt = []

for i in range(16):
    if int(df.iloc[i, 2] == 1):
        AI_snt += sent_tokenize(df.iloc[i, 0])
    elif int(df.iloc[i, 2] == 0):
        Human_snt += sent_tokenize(df.iloc[i, 0])

print(AI_snt)
print(Human_snt)



#GPU Setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current Device : {device}")

Model_name = 'bert-base-cased'
model = AutoModelForMaskedLM.from_pretrained(Model_name)
tokenizer = AutoTokenizer.from_pretrained(Model_name)

mask_fill = pipeline(task = 'fill-mask', model=model, tokenizer=tokenizer, device=device)

result_df = pd.DataFrame({'p_img' : [], 'label' : []})
print(result_df)
#bert-base-multilingual-cased 모델은 BertTokenizerFast class를 사용.
#tokenizer.encode : 정수형 토큰 리스트로 encoding
#tokenizer.decode : 정수형 토큰 리스트를 string으로 decode. skip_special_token 을 쓰면 cls, sep이 삭제됨.
#주의 : skip_special_token : [unk], [MASK] 도 삭제됨.

def masking(index : int, token_li : list) -> list:
    result = token_li[::]
    result[index] = 103
    return result

#batch : 한 번에 확률을 구할 토큰의 개수. -> 글의 어떤 부분에서 잘리던지 신경쓰진 않는다.
#batch를 잡아 시간복잡도를 감축하고, 데이터 증강의 효과를 본다.
def make_data(string : str, batch : int) -> list:
    #특수 토큰 : [CLS] : 101, [SEP] : 102, [MASK] : 103, [UNK] : 100 으로 추정됨.
    token_list = tokenizer.encode(string)
    token_num = len(token_list)

    if token_num == batch:
        size = batch
        pass

    #토큰 개수가 지정폭 (64)보다 많은 경우
    elif token_num > batch:
        size = batch
        #cls와 sep는 없어도 다시 원문으로 되돌릴 수는 있음
        token_list = token_list[0:batch]

    #토큰 개수가 지정폭 보다 적은 경우 but 최소폭 보다는 큼
    #at first 16 -> update to 24
    elif token_num >= int(batch/3):
        size = token_num

    #너무 짧은 문장 드롭
    else:
        return False


    #토큰 개수 기준으로 자른 문장 출력.

    #return 값을 저장할 리스트. 편의상 Padding은 -1로 했다고 보면 된다.
    res_list = [[-1]*(batch) for _ in range(batch)]
    #res_list[i][j] : token_list의 i번 인덱스 까지를 사용, j번 인덱스에 Mask 씌움
    #이때 i > j 가 값이 존재할 (-1)이 아닐 조건이다.

    for i in range(0, size):
        for j in range(0, i):
            tok_li = token_list[0:i+1]
            answer_token = tok_li[j]
            tok_to_pipe = masking(j, tok_li)
            ptm_res = mask_fill(tokenizer.decode(tok_to_pipe))
            is_in_top_k = False
            for res in ptm_res:
                if res['token'] == answer_token:
                    res_list[i][j] = res['score']
                    is_in_top_k = True
                    break
            if(is_in_top_k == False):
                res_list[i][j] = 0
    
    return res_list

print(len(AI_snt), len(Human_snt))


img_arr = []
for string in AI_snt:
    img = make_data(string, 64)
    #너무 짧은 경우
    if img == False:
        continue
    df2 = pd.DataFrame({"p_img" : [img], "label" : [1]})
    result_df = pd.concat([result_df, df2], ignore_index = True)


"""
img_arr = []
for string in Human_snt:
    img = make_data(string, 64)
    #너무 짧은 경우
    if img == False:
        continue
    df2 = pd.DataFrame({"p_img" : [img], "label" : [0]})
    result_df = pd.concat([result_df, df2], ignore_index = True)

"""
result_df.to_csv("BenchMark/bench_data_onlyai.csv")
print("Task Done")

#Nvida Geforce RTX 3060 기준 64*64 -> 30초, 32*32 -> 5초 정도 소요

#몇가지 이슈가 있음
#Note : batch 크기를 32로 할 경우 문장을 부분을 정확히 추출하는 작업이 선행되어야 함
#Note : 확률 계산을 모든 단어가 아니라 2개, 3개씩 건너뛰며 할 수 있음. (CNN에서 차용한 아이디어)
#Note : 과연 몇개까지 데이터를 얻어야 하는가....?