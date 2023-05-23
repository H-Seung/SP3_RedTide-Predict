from flask import Flask, render_template, request
import pickle
from datetime import datetime
import numpy as np
from flask_app import api_get

thres = 0.74

# model 가져오기
model = None
with open('model_v3.pkl','rb') as pickle_file:
    model = pickle.load(pickle_file)

# xtestpred 데이터 가져오기
with open('xtestpred.pkl','rb') as pickle_file:
    xtestpred = pickle.load(pickle_file)

# api 예보데이터 가져오기
df_predict = api_get.df_predict



# 예보 데이터 df_predict 에 훈련모델 적용
y_pred_proba = model.predict_proba(df_predict)[:,-1]

# 예보 데이터에 대한 적조발생여부 예측 확률과 예측 결과값을 한 표로 정리
result = df_predict.copy()
result['pred_prob'] = y_pred_proba
result['pred_result'] = np.where(result['pred_prob'] < thres, 0, 1) # 임계값 thres 보다 작을 경우 0, 이상일 경우 1


# 예측한 결과값인 0 또는 1에 해당할 확률을 같이 나타내주기 위해,
# test 데이터셋으로 0값의 분포, 1값의 분포 생성
pred_0 = xtestpred.loc[xtestpred['pred_result'] == 0, 'pred_prob'].sort_values(ascending=True)
pred_1 = xtestpred.loc[xtestpred['pred_result'] == 1, 'pred_prob'].sort_values(ascending=True)

# 실제값(proba)을 입력하면 분포에서 해당하는 위치를 %로 출력하는 함수
def get_percentiles(proba, thres=thres):
    """proba: 확인할 실제 proba 값, thres: 0과 1을 나누는 proba 임계값
    """
    if proba <= 0.000001:
        percentiles = 0.999999
    elif proba >= thres: # 1일 확률
        percentiles = np.interp(proba, pred_1, np.arange(sum(xtestpred['pred_result'] == 1))/sum(xtestpred['pred_result'] == 1))
    else:               # 0일 확률
        percentiles = np.interp(proba, pred_0, np.arange(sum(xtestpred['pred_result'] == 0))/sum(xtestpred['pred_result'] == 0))
        percentiles = 1 - percentiles
    return percentiles

# 예보데이터 pred_prob 을 이용해 0일 확률, 1일 확률 계산
for i, row in result.iterrows():
    result.at[i, 'result_prob'] = get_percentiles(row['pred_prob'])



# -------------------------------------------------------------------
# flask 실행
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

