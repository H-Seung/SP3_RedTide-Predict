from flask import Flask, render_template, request
import requests
import json
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
import api_get

# model 가져오기
model = None
with open('../model.pkl','rb') as pickle_file:
    model = pickle.load(pickle_file)

# X_train 데이터 가져오기
with open('../X_train.pkl','rb') as pickle_file:
    X_train = pickle.load(pickle_file)

df_predict = api_get.df_predict


# 실제값 입력하면 분포(array)에서 해당하는 위치를 %로 출력
def value_to_percentile(array, value):
    array = np.sort(array)
    index = np.searchsorted(array, value, side='left')
    if index == 0:
        return 0.0
    elif index == len(array):
        return 100.0
    else:
        return (index - 1) / len(array) * 100

# 0값 proba의 분포, 1값 proba의 분포
x_pred = X_train.copy()
x_pred['pred'] = model.predict_proba(X_train)[:,-1]

proba_0 = x_pred[x_pred['pred']<0.24].pred.sort_values()
proba_1 = x_pred[x_pred['pred']>=0.24].pred.sort_values()


# 예측 데이터프레임 생성
result = df_predict.copy()
result['pred'] = model.predict_proba(df_predict)[:,-1]

# 임계값 0.24보다 작을 경우 0, 이상일 경우 1 
result['pred_result'] = np.where(result['pred'] < 0.24, 0, 1)

# 0 또는 1일 확률 제공
for i, row in result.iterrows():
    x = row['pred_result']
    if x == 0:
        result.at[i, 'prob'] = 100-(value_to_percentile(array=proba_0, value=row['pred']))
    elif x == 1:
        result.at[i, 'prob'] = value_to_percentile(array=proba_1, value=row['pred'])


# -----------------------------------------------------------
app = Flask(__name__)

@app.route('/')
def main():
    today = datetime.now().strftime('%Y-%m-%d')
    return render_template('main.html', today=today, df_predict=[df_predict.to_html(classes='data')], titles=df_predict.columns.values) # render_template()은 template이란 폴더내에서 자동으로 가져옴

@app.route('/predict', methods=['POST'])
def post(): # 들어오고나서
    day_input = request.form['input'] # html의 name='input'
    day_input = int(day_input)
    output = result.loc[day_input]
    return  render_template('predict.html', data=output, day=day_input)


if __name__=='__main__':
    app.run(debug=True)

