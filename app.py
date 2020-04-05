#!/usr/bin/python
#-*-coding: utf-8 -*-
##from __future__ import absolute_import
###
from flask import Flask, jsonify, render_template, request
import json
import numpy as np
import pandas as pd
from pythainlp import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize

from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,TemplateSendMessage,ImageSendMessage, StickerSendMessage, AudioSendMessage
)
from linebot.models.template import *
from linebot import (
    LineBotApi, WebhookHandler
)

app = Flask(__name__)

lineaccesstoken = 'S8q+jn6JQOml7H7w91vcHd1sVVlyuLxKHOb/PPQFrmo+2LX8sx+um9w6ei+NvxB996hYf9JZ8VftARif9bKfzUeR6IKZqNk4XUdsIQWoXJVEJEg9X9HnIWSJ7FCfCztrMUt1yeAXilQktgT4vt61HQdB04t89/1O/w1cDnyilFU='
line_bot_api = LineBotApi(lineaccesstoken)

####################### new ########################
@app.route('/')
def index():
    return "Hello World!"


@app.route('/webhook', methods=['POST'])
def callback():
    json_line = request.get_json(force=False,cache=False)
    json_line = json.dumps(json_line)
    decoded = json.loads(json_line)
    no_event = len(decoded['events'])
    for i in range(no_event):
        event = decoded['events'][i]
        event_handle(event)
    return '',200


def event_handle(event):
    print(event)
    try:
        userId = event['source']['userId']
    except:
        print('error cannot get userId')
        return ''

    try:
        rtoken = event['replyToken']
    except:
        print('error cannot get rtoken')
        return ''
    try:
        msgId = event["message"]["id"]
        msgType = event["message"]["type"]
    except:
        print('error cannot get msgID, and msgType')
        sk_id = np.random.randint(1,17)
        replyObj = StickerSendMessage(package_id=str(1),sticker_id=str(sk_id))
        line_bot_api.reply_message(rtoken, replyObj)
        return ''

    if msgType == "text":
        msg = str(GetResult(event["message"]["text"]))
        replyObj = TextSendMessage(text=msg)
        line_bot_api.reply_message(rtoken, replyObj)

    else:
        sk_id = np.random.randint(1,17)
        replyObj = StickerSendMessage(package_id=str(1),sticker_id=str(sk_id))
        line_bot_api.reply_message(rtoken, replyObj)
    return ''

if __name__ == '__main__':
    app.run(debug=True)

def splittraintest(dat,trainratio=0.7):
    sdat = dat.sample(frac=1,random_state=0)
    ntrain = int(len(dat)*trainratio)
    traindat = sdat.iloc[0:ntrain]
    testdat = sdat.iloc[ntrain:]
    return traindat,testdat

def get_wo_cr_tokens(text):
    return word_tokenize(text) + list(text)

def TrainModel(dat):
    trdat,tedat = splittraintest(dat)

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

    return model

def GetResult(keyword):
    xl = pd.ExcelFile('IT-KMITL-dataset.xlsx')
    dat = xl.parse(sheet_name='Sheet1')
    ansdat = xl.parse(sheet_name='Sheet2')

    model = TrainModel(dat)

    keyword = [keyword]
    feat = vectorizer.transform(keyword)
    feat_norm = normalize(feat)
    intent = model.predict(feat_norm)[0]

    result = ansdat[ansdat['Intent'] == intent]
    textresult = result
    textresult = textresult['Answer'].values+'\n'+textresult['Source'].values
    textresult = textresult[0]
    return textresult
