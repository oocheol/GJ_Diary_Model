from flask import Flask
from flask import request
import pandas as pd
from flask_cors import CORS
import pickle
app = Flask(__name__)
# 내가 만든 모델을 "test.pkl"이란 파일에 저장, 해당 모델을 불러온다
test = pickle.load(open('test.pkl', 'rb'))
#POST 방식으로 값을 불러올경우 인코딩 과정 없이 받게 해줌
CORS(app)
@app.route("/", methods = ["GET", "POST"])
def connect():
    value="hello"
    if request.method == 'POST':
        train_tes = []
        #POST 형식으로 전송된 값을 value에 저장
        value = dict(request.form)
        #저장된 값을 DataFrame으로 변환한다.
        df = pd.DataFrame(value, index=[0])
        #pickle을 통해 불러온 모델에 불러온 값을 학습시켜 모델 값을 받는다.
        value = str(int(test.predict(df)))
        print(value)
    #학습시킨 값을 return.
    return value
        
if __name__ == "__main__" :
    app.run()