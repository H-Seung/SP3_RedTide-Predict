# 오픈 API 를 이용해 예보 데이터를 가져와 모델이 예측할 수 있는 형태로 가공
import os
from dotenv import load_dotenv
import pandas as pd
import requests
import json
from datetime import datetime, timedelta
from pytz import timezone

# import os
# parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
# print(parent_dir) # 부모 디렉터리의 절대경로 얻기

dotenv_path = os.path.abspath("../ENVRONS.env")
load_dotenv(dotenv_path=dotenv_path)
API_KEY = os.getenv("API_KEY")


# 예보 조회날짜 설정
t = datetime.now(timezone('Asia/Seoul'))
print("t:",t)
if t.hour < 7: # 7시 전이라면 1일 전 예보(예보는 6시에 게시되지만 여유있게 7로 설정)
    today = (t-timedelta(days=1)).strftime('%Y%m%d')+'0600'
else:
    today = t.strftime('%Y%m%d')+'0600'
print("조회날짜(금일):", today)

## 해상날씨정보 (파고) -----------------------------
url_sea = 'https://apis.data.go.kr/1360000/MidFcstInfoService/getMidSeaFcst'
params_sea ={'serviceKey' : API_KEY, 
'dataType' : 'json', 
'pageNo' : '1', 
'numOfRows' : '10', 
'regId' : '12B10000',
'tmFc' : today }

resp_sea = requests.get(url_sea, params=params_sea, verify=False)
data_sea_list = json.loads(resp_sea.text)['response']['body']['items']['item']
df_sea = pd.json_normalize(data_sea_list)

# 사용할 데이터만 추출,가공
df_sea_e = pd.DataFrame()
df_sea_e['whHigh3'] = (df_sea.iloc[:,16] + df_sea.iloc[:,17])/2
df_sea_e['whHigh4'] = (df_sea.iloc[:,20] + df_sea.iloc[:,21])/2
df_sea_e['whHigh5'] = (df_sea.iloc[:,24] + df_sea.iloc[:,25])/2
df_sea_e['whHigh6'] = (df_sea.iloc[:,28] + df_sea.iloc[:,29])/2
df_sea_e['whHigh7'] = (df_sea.iloc[:,32] + df_sea.iloc[:,33])/2
df_sea_e['whHigh8'] = df_sea.iloc[:,35]
df_sea_e['whHigh9'] = df_sea.iloc[:,37]
df_sea_e['whHigh10'] = df_sea.iloc[:,39]
df_sea_e['whAvg3'] = (((df_sea.iloc[:,14] + df_sea.iloc[:,15])/2) + df_sea_e['whHigh3'])/2
df_sea_e['whAvg4'] = (((df_sea.iloc[:,18] + df_sea.iloc[:,19])/2) + df_sea_e['whHigh3'])/2
df_sea_e['whAvg5'] = (((df_sea.iloc[:,22] + df_sea.iloc[:,23])/2) + df_sea_e['whHigh3'])/2
df_sea_e['whAvg6'] = (((df_sea.iloc[:,26] + df_sea.iloc[:,27])/2) + df_sea_e['whHigh3'])/2
df_sea_e['whAvg7'] = (((df_sea.iloc[:,30] + df_sea.iloc[:,31])/2) + df_sea_e['whHigh3'])/2
df_sea_e['whAvg8'] = (df_sea.iloc[:,34] + df_sea.iloc[:,35])/2
df_sea_e['whAvg9'] = (df_sea.iloc[:,36] + df_sea.iloc[:,37])/2
df_sea_e['whAvg10'] = (df_sea.iloc[:,38] + df_sea.iloc[:,39])/2

col_order= ['whAvg3', 'whHigh3', 'whAvg4', 'whHigh4', 'whAvg5', 'whHigh5', 'whAvg6', 'whHigh6', 
            'whAvg7', 'whHigh7', 'whAvg8', 'whHigh8', 'whAvg9', 'whHigh9', 'whAvg10', 'whHigh10']
df_sea_e = df_sea_e[col_order]


## 기온정보 ----------------------------------------
url_t = 'https://apis.data.go.kr/1360000/MidFcstInfoService/getMidTa?'
params_t ={'serviceKey' : API_KEY, 
'dataType' : 'json', 
'pageNo' : '1', 
'numOfRows' : '10', 
'regId' : '11F20401',
'tmFc' : today }

resp_t = requests.get(url_t, params=params_t, verify=False)
data_t_list = json.loads(resp_t.text)['response']['body']['items']['item']
df_t = pd.json_normalize(data_t_list)

# 사용할 데이터만 추출,가공
df_t_e = df_t.loc[:,['taMin3','taMax3','taMin4','taMax4','taMin5','taMax5','taMin6','taMax6','taMin7','taMax7','taMin8','taMax8','taMin9','taMax9','taMin10','taMax10']]
df_t_e['taAvg3'] = (df_t_e['taMin3'] + df_t_e['taMax3']) / 2
df_t_e['taAvg4'] = (df_t_e['taMin4'] + df_t_e['taMax4']) / 2
df_t_e['taAvg5'] = (df_t_e['taMin5'] + df_t_e['taMax5']) / 2
df_t_e['taAvg6'] = (df_t_e['taMin6'] + df_t_e['taMax6']) / 2
df_t_e['taAvg7'] = (df_t_e['taMin7'] + df_t_e['taMax7']) / 2
df_t_e['taAvg8'] = (df_t_e['taMin8'] + df_t_e['taMax8']) / 2
df_t_e['taAvg9'] = (df_t_e['taMin9'] + df_t_e['taMax9']) / 2
df_t_e['taAvg10'] = (df_t_e['taMin10'] + df_t_e['taMax10']) / 2
col_order = ['taAvg3', 'taMin3','taMax3','taAvg4', 'taMin4','taMax4','taAvg5', 'taMin5','taMax5','taAvg6', 'taMin6','taMax6',
             'taAvg7', 'taMin7','taMax7','taAvg8', 'taMin8','taMax8','taAvg9', 'taMin9','taMax9','taAvg10', 'taMin10','taMax10']
df_t_e = df_t_e[col_order]


## 육상날씨정보 (강수확률) ---------------------------
url_land = 'https://apis.data.go.kr/1360000/MidFcstInfoService/getMidLandFcst'
params_land ={'serviceKey' : API_KEY, 
'dataType' : 'json', 
'pageNo' : '1', 
'numOfRows' : '10', 
'regId' : '11F20000',
'tmFc' : today }

resp_land = requests.get(url_land, params=params_land, verify=False)
data_land_list = json.loads(resp_land.text)['response']['body']['items']['item']
df_land = pd.json_normalize(data_land_list)

# 사용할 데이터만 추출,가공
df_land_e = pd.DataFrame()
df_land_e['rain3'] = (df_land['rnSt3Am'] + df_land['rnSt3Pm']) / 2
df_land_e['rain4'] = (df_land['rnSt4Am'] + df_land['rnSt4Pm']) / 2
df_land_e['rain5'] = (df_land['rnSt5Am'] + df_land['rnSt5Pm']) / 2
df_land_e['rain6'] = (df_land['rnSt6Am'] + df_land['rnSt6Pm']) / 2
df_land_e['rain7'] = (df_land['rnSt7Am'] + df_land['rnSt7Pm']) / 2
df_land_e['rain8'] = df_land['rnSt8']
df_land_e['rain9'] = df_land['rnSt9']
df_land_e['rain10'] = df_land['rnSt10']

# 과거 데이터는 Rainfall 강수량으로 들어가있으나, 미래데이터는 강수확률이므로
# 강수량과 동일한 수준으로 만들기 위해 계수 0.02를 곱합(ex. 30(%) -> 0.6(mm))
df_land_e = df_land_e * 0.02


# 미래데이터(api)를 학습데이터 구조와 동일하게 생성
## avg_Temp, low_Temp, high_Temp, Rainfall, avg_WaveHeight, high_WaveHeight
col_name = ['day','whAvg', 'whHigh', 'taAvg', 'taMin', 'taMax', 'rain']
df_predict = pd.DataFrame(columns=col_name)
for i in range(0,8):
    df_predict.loc[i,'day'] = i+3  # 예측하려는 일수
    df_predict.iloc[i,1:] = pd.concat([df_sea_e.filter(regex="{}".format(i+3)), 
                            df_t_e.filter(regex="{}".format(i+3)), 
                            df_land_e.filter(regex="{}".format(i+3))], 
                            axis=1)
df_predict = df_predict[['day','taAvg', 'taMin', 'taMax', 'rain', 'whAvg', 'whHigh']] # 컬럼 순서변경
df_predict = df_predict.set_index('day')

print(df_predict)