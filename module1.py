import numpy as np
import pandas as pd
from pythainlp import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize

def splittraintest(dat,trainratio=0.7):
    sdat = dat.sample(frac=1,random_state=0)
    ntrain = int(len(dat)*trainratio)
    traindat = sdat.iloc[0:ntrain]
    testdat = sdat.iloc[ntrain:]
    return traindat,testdat

def get_wo_cr_tokens(text):
    return word_tokenize(text) + list(text)

def trainModel(dat):
    trdat,tedat = splittraintest(dat)

    trkeyword = trdat['Keyword'].values
    vectorizer = TfidfVectorizer(tokenizer=get_wo_cr_tokens, ngram_range=(1,3))
    vectorizer.fit(trkeyword)

    tekeyword = tedat['Keyword'].values
    trfeat = vectorizer.transform(trkeyword)
    tefeat = vectorizer.transform(tekeyword)

    trlabel = trdat['Intent'].values
    telabel = tedat['Intent'].values
    trfeat_norm = normalize(trfeat)
    tefeat_norm = normalize(tefeat)
    model = LinearSVC(random_state=0)
    model.fit(trfeat_norm, trlabel)

    return model, vectorizer

def getresult(keyword):
    xl = pd.ExcelFile('IT-KMITL-dataset.xlsx')
    dat = xl.parse(sheet_name='Sheet1')
    ansdat = xl.parse(sheet_name='Sheet2')

    model, vectorizer = trainModel(dat)

    keyword = [keyword]
    feat = vectorizer.transform(keyword)
    feat_norm = normalize(feat)
    intent = model.predict(feat_norm)[0]

    result = ansdat[ansdat['Intent'] == intent]
    textresult = result
    
    if textresult.empty:
        return print"ขออภัยยังไม่สามารถตอบคำถามนี้ได้ค่ะ"
    else:
        textresult = textresult['Answer'].values+'\n'+textresult['Source'].values
        textresult = textresult[0]
        return str(textresult)
        
    
