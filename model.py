import pandas as pd
import numpy as np
import pymysql
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt
import pickle

# mysql 연결
def mysql_connect():
    conn=pymysql.connect(host='localhost',
                        user='root',
                        password='dkzndkakfls!0',
                        db='pj3',
                        charset='utf8',
                        cursorclass=pymysql.cursors.DictCursor) # dict 형식
    cur=conn.cursor()
    sql = "select * from pj3.data"
    cur.execute(sql)
    output = cur.fetchall()

    conn.close()
    return output


# 데이터프레임 형태로 data 가져옴
data = pd.DataFrame(mysql_connect())

# 훈련,검증,테스트셋 분리
df=data.copy()

target = 'YN'

train, test = train_test_split(df, train_size=0.80, test_size=0.20, 
                              stratify=df[target], random_state=2)
train, val = train_test_split(train, train_size=0.80, test_size=0.20, 
                              stratify=train[target], random_state=2)
print(f"train set : {train.shape} \nval. set : {val.shape} \ntest set : {test.shape}")

# 특성, 타겟 분리
features = train.drop(columns=[target, 'Date']).columns
X_train = train[features]
y_train = train[target]
X_val = val[features]
y_val = val[target]
X_test = test[features]
y_test = test[target]


# 성능지표 결과 출력
def show_result(my_pipe, X_data = X_val, y_data = y_val, threshold = 0.5):
    y_pred_proba = my_pipe.predict_proba(X_data)[:,-1]
    y_pred = y_pred_proba >= threshold
    print('Precision Score: ',precision_score(y_data, y_pred).round(4))
    print('Recall Score: ',recall_score(y_data, y_pred).round(4))
    print('F1 Score:',f1_score(y_data, y_pred).round(4))
    print('Accuracy Score:',accuracy_score(y_data, y_pred).round(4))
    
    y_pred_proba = my_pipe.predict_proba(X_data)[:,-1]
    print('AUC Score:', roc_auc_score(y_data, y_pred_proba).round(4))

# roc curve & 임계값별 f1 score
def draw_roc(my_pipe):
  y_pred_proba = my_pipe.predict_proba(X_val)[:,-1]
  threshold = np.arange(0.1,0.55,0.01)
  fprs, tprs, th = roc_curve(y_val, y_pred_proba)
  plt.plot(fprs , tprs, label='ROC')
  plt.title('ROC curve')
  plt.xlabel('FPR(Fall-out)')
  plt.ylabel('TPR(Recall)')
  plt.show();

  for t in threshold:
    y_pred = y_pred_proba >= t
    val_f1 = round(f1_score(y_val, y_pred),4)
    val_roc = round(roc_auc_score(y_val, y_pred_proba),4)
    print(round(t,2), val_f1, val_roc)
  return fprs, tprs, th



# 기준모델
print("----------------기준모델(Logistic)----------------")
logis = LogisticRegression(random_state=42)
logis.fit(X_train, y_train)

print('훈련 Accuracy Score:',logis.score(X_train, y_train).round(4))
y_pred = logis.predict(X_val)
print('검증 Accuracy Score:',accuracy_score(y_val, y_pred).round(4))
print('검증 F1 Score:',f1_score(y_val, y_pred).round(4))
y_pred_proba = logis.predict_proba(X_val)[:,-1]
print('검증 AUC Score:', roc_auc_score(y_val, y_pred_proba).round(4))
print(classification_report(y_val, y_pred))
print("-------------------------------------------------")


# 모델 학습
print("---------------- RandomForest -------------------")
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
    cv = 3,
    scoring = 'roc_auc',
    verbose = 1,
    n_jobs = -1,
    random_state = 42
)
clf.fit(X_train,y_train)
print('최적 하이퍼파라미터: ', clf.best_params_)
print('auc : ', clf.best_score_)

best_model = clf.best_estimator_ # 최적의 파라미터 조합이 적용된 모델

print("최적의 임계값 찾기")
print("threshold", "f1", "roc")
fpr, tpr, th = draw_roc(best_model)

print("\n--- threshold=0.24 일 때 성능 ---")
show_result(best_model, threshold=0.24)
print("---------------------------------------------------")


print("---------------- 일반화 성능 ----------------------")
y_test = test[target]
show_result(best_model, X_data = X_test, y_data = y_test, threshold = 0.24)
print("--------------------------------------------------")


with open('model.pkl', 'wb') as pickle_file:
    pickle.dump(best_model, pickle_file)
    
with open('X_train.pkl', 'wb') as pickle_file:
    pickle.dump(X_train, pickle_file)
    