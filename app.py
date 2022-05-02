from flask import Flask
from flask import request
import pandas as pd
from flask_cors import CORS
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import re
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
# 내가 만든 모델을 "test.pkl"이란 파일에 저장, 해당 모델을 불러온다
# test = pickle.load(open('test.pkl', 'rb'))
#POST 방식으로 값을 불러올경우 인코딩 과정 없이 받게 해줌
CORS(app)
# model = keras.models.load_model("model.h5")
loaded_model = load_model('best_model.h5')
okt = Okt()
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
tokenizer = Tokenizer()
max_len = 30

def sentiment_predict(new_sentence):
  new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', new_sentence)
  new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
  new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
  encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
  pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
  score = float(loaded_model.predict(pad_new)) # 예측
  return "긍정 {:.2f}% / 부정 {:.2f}%\n".format(score * 100,(1- score) * 100)

@app.route("/", methods = ["GET", "POST"])
def connect():
    value="안녕하세요"
    if request.method == 'POST':
        train_tes = []
        #POST 형식으로 전송된 값을 value에 저장
        value = dict(request.form)
        #저장된 값을 DataFrame으로 변환한다.
        df = pd.DataFrame(value, index=[0])
        #pickle을 통해 불러온 모델에 불러온 값을 학습시켜 모델 값을 받는다.
        # value = str(int(test.predict(df)))
        value = sentiment_predict(value)
    #학습시킨 값을 return.
    return value
        
if __name__ == "__main__" :
    app.run()