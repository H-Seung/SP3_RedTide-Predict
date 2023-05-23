# db에서 훈련데이터를 가져와 모델을 구축해 훈련시킨 후 pkl로 내보내기 및 성능평가를 진행한다.
# 추가적으로 예측에서 사용할 데이터 분포를 pkl로 내보낸다.
import os
from dotenv import load_dotenv
import pymysql
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import matplotlib.pyplot as plt
import pickle

dotenv_path = os.path.abspath("../ENVRONS.env")
load_dotenv(dotenv_path=dotenv_path)
DB_ID = os.getenv("DB_ID")
DB_PW = os.getenv("DB_PW")


# 데이터베이스 연결
db_connection = create_engine(f'mysql+pymysql://{DB_ID}:{DB_PW}@localhost/pj3', echo=True)
conn = db_connection.connect()

# 저장되어 있는 DB data 가져오기
df = pd.read_sql_table('dataset', con=conn)
conn.close()

# # DB 가 아닌 csv에서 data를 가져오고자 하는 경우
# df = pd.read_csv("Data_3_oversampling.csv")


# 특성, 타겟 분리
target = 'YN'
features = ['taAvg', 'taMin', 'taMax', 'rain', 'whAvg', 'whHigh']
X = df[features]
y = df[target]

# 훈련셋, 테스트셋 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, 
                              shuffle=True, random_state=42)
print(f"X_train : {X_train.shape} \nX_test : {X_test.shape}")
print(f"y_train : {y_train.shape} \ny_test : {y_test.shape}")


# 성능지표 결과 출력 함수 정의
def show_result(my_pipe, X_data, y_data, threshold = 0.5):
    y_pred_proba = my_pipe.predict_proba(X_data)[:,-1]
    y_pred = y_pred_proba >= threshold
    print('Precision Score: ',precision_score(y_data, y_pred).round(6))
    print('Recall Score: ',recall_score(y_data, y_pred).round(6))
    print('F1 Score:',f1_score(y_data, y_pred).round(6))
    print('Accuracy Score:',accuracy_score(y_data, y_pred).round(6))
    
    y_pred_proba = my_pipe.predict_proba(X_data)[:,-1]
    print('AUC Score:', roc_auc_score(y_data, y_pred_proba).round(6))
    
    return y_pred, y_pred_proba


# roc curve 그래프 & 임계값별 f1 score 출력 함수 정의
def draw_roc(my_pipe, X_data, y_data):
    y_pred_proba = my_pipe.predict_proba(X_data)[:,-1]
    threshold = np.arange(0.35,0.85,0.01)
    fprs, tprs, th = roc_curve(y_data, y_pred_proba)
    plt.plot(fprs , tprs, label='ROC')
    plt.title('ROC curve')
    plt.xlabel('FPR(Fall-out)')
    plt.ylabel('TPR(Recall)');

    for t in threshold:
        y_pred = y_pred_proba >= t
        val_f1 = round(f1_score(y_data, y_pred),6)
        val_roc = round(roc_auc_score(y_data, y_pred_proba),6)
        print(round(t,2), val_f1, val_roc)

    return fprs, tprs, th


# 기준모델 (로지스틱 회귀모델)
print("\n----------------기준모델(Logistic)----------------")
logis = LogisticRegression(random_state=42)
k = 5

scores_accuracy = cross_val_score(logis, X_train, y_train, cv=k, scoring='accuracy')
scores_f1 = cross_val_score(logis, X_train, y_train, cv=k, scoring='f1_macro')
scores_auc = cross_val_score(logis, X_train, y_train, cv=k, scoring='roc_auc')

print(f"Accuracy: {scores_accuracy.mean():.6f} +/- {scores_accuracy.std():.6f}")
print(f"F1 Score: {scores_f1.mean():.6f} +/- {scores_f1.std():.6f}")
print(f"AUC Score: {scores_auc.mean():.6f} +/- {scores_auc.std():.6f}")
print("-------------------------------------------------")


# 모델 학습
print("\n---------------- RandomForest -------------------")
rf = RandomForestClassifier(random_state=42)

dists = {
    'n_estimators': randint(50, 400), 
    'max_depth' : [None,5,10,15,20],
    'max_samples' : [None,0.2,0.4,0.7,1.0],
    'min_samples_leaf' : [1,3,6,9,15]
}

clf = RandomizedSearchCV(
    rf,
    param_distributions = dists,
    n_iter = 30,
    cv = 5,
    scoring = 'roc_auc',
    verbose = 1,
    n_jobs = -1,
    random_state = 42
)
clf.fit(X_train,y_train)

print('최적 하이퍼파라미터: ', clf.best_params_)
print('auc : ', clf.best_score_)

best_model = clf.best_estimator_ # 최적의 파라미터 조합이 적용된 모델 저장


# pickle 로 훈련한 모델 내보내기
with open('model_v3.pkl', 'wb') as pickle_file:
    pickle.dump(best_model, pickle_file)


# 성능 평가
print("\n전체 train set - 최적의 임계값 찾기")
print("threshold", "f1", "roc")
fpr, tpr, th = draw_roc(best_model, X_data=X_train, y_data=y_train)

thres = 0.55
print(f"\n------ 전체 train set 에서의 성능 (thres={thres}) ------")
show_result(best_model, X_data=X_train, y_data=y_train, threshold=thres)
print("-----------------------------------------------------")


print("\n최종 test set - 최적의 임계값 찾기")
print("threshold", "f1", "roc")
fpr, tpr, th = draw_roc(best_model, X_data=X_test, y_data=y_test)

thres = 0.74
print(f"\n------ 최종 test set 에서의 성능 (thres={thres}) ------")
pred_test, proba_test = show_result(best_model, X_data=X_test, y_data=y_test, threshold=thres)
print("----------------------------------------------------")


# 예측(flask_app)에서 0/1일 확률을 계산하는 데 사용될 데이터분포 생성
xtestpred = X_test.copy()
xtestpred['pred_prob'] = proba_test
xtestpred['pred_result'] = pred_test.astype(int) # bool -> int

# 데이터분포 내보내기
with open('xtestpred.pkl', 'wb') as pickle_file:
    pickle.dump(xtestpred, pickle_file)
