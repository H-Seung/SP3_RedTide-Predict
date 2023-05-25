# 적조현상 발생여부 예측 모델
## 디렉터리 구조
```
.
|-- AI_16_한승희_Section3_v3.ipynb
|-- Data_1_preprocess.csv
|-- Data_2_FE.csv
|-- Data_3_oversampling.csv
|-- README.md
`-- project3_files
    |-- docker-compose.yaml
    |-- flask_app
    |   |-- __init__.py
    |   |-- api_get.py
    |   `-- templates
    |       |-- main.html
    |       `-- predict.html
    |-- model.pkl
    |-- model.py
    |-- model_v3.pkl
    |-- requirements_v3.txt
    `-- xtestpred.pkl
```

## 파일 설명
- `AI_16_한승희_Section3.ipynb` : 프로젝트 전체 과정을 ipynb 파일로 정리
  - 학습데이터 가공 및 EDA 수행, DB 저장 
  - 모델구축(`model.py` 와 동일)
  - 예측(`flask_app/` 와 동일)
- `Data_1_preprocess.csv`,`Data_2_FE.csv`,`Data_3_oversampling.csv` : 각 전처리, 특성공학, 오버샘플링 단계를 마친 데이터셋 파일
- `project3_files/model.py` : DB 또는 csv 파일에서 data를 불러와 모델 학습 및 최종 모델을 내보냄
- `project3_files/model_v3.pkl` : 최종 모델 pkl 파일
- `project3_files/flask_app/` : API를 이용해 데이터를 수집하고 모델에 적용하여 예측 결과를 웹서비스로 제공
- `project3_files/xtestpred.pkl` : 예측에서 0 또는 1일 확률을 계산하는 데 사용될 데이터분포

## 프로젝트 배경
- 매년 적조로 인한 수산업 피해가 발생하는 가운데, 공공데이터를 활용하여 적조 발생여부를 예측하는 모델을 마련하고자 함

## 프로젝트 개요
- 기상청 등 공공데이터와 API를 활용하여 랜덤포레스트로 적조현상 발생여부 예측

## 프로젝트 절차
![image](https://github.com/H-Seung/SP3_RedTide-Predict/assets/114974542/7de4734a-67b2-4620-ba1c-3bdb558694aa)

### 모델 훈련 및 구축
- 모델은 로지스틱 회귀모델과 랜덤포레스트 두 모델을 테스트하여 더 높은 성능을 보이는 랜덤포레스트 모델을 사용
- 최적 하이퍼파라미터:  {'max_depth': 20, 'max_samples': None, 'min_samples_leaf': 1, 'n_estimators': 376}
- 일반화 성능 :
  ```
  Precision Score:  0.984874
  Recall Score:  0.971808
  F1 Score: 0.978297
  Accuracy Score: 0.97853
  AUC Score: 0.99769
  ```

### MySQL 데이터 저장
- 최종 가공된 훈련데이터를 DB에 저장
![image](https://github.com/H-Seung/SP3_RedTide-Predict/assets/114974542/59690607-0da8-4ee0-9ecf-187130edb64f)
### flask 이용한 웹페이지 구현
- 기상청으로부터 3~10일 후의 해상,육상 날씨예보를 API로 가져와 가공후 모델 적용
- 오늘 날짜를 기준으로 3~10 사이의 숫자를 입력하면 해당 일수 뒤의 적조 발생여부를 예측   

![image](https://user-images.githubusercontent.com/114974542/229387655-20ff65b9-9b77-48b1-ba1c-d3a4dbb9fdb7.png)
![image](https://user-images.githubusercontent.com/114974542/229387661-4daaf3c5-18b8-49dc-abf1-62af5cdc01ac.png)
