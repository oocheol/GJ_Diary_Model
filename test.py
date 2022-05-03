from tensorflow.keras.models import load_model
import re
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv

X_train = []
f = open("X_train.csv",'r')
rea = csv.reader(f)
for row in rea:
    X_train.append(row)
f.close

# model = keras.models.load_model("model.h5")
okt = Okt()
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
# tokenizer = Tokenizer()
max_len = 30

tokenizer = Tokenizer(19416)
tokenizer.fit_on_texts(X_train)

loaded_model = load_model('best_model.h5')

def sentiment_predict(new_sentence):
  new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', new_sentence)
  new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
  new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
  encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
  pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
  score = float(loaded_model.predict(pad_new)) # 예측
  print("긍정 {:.2f}% / 부정 {:.2f}\n".format(score * 100,(1- score) * 100))