import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import time

train = pd.read_csv(r"E:\Hackathon\Janatahack Cross-sell Prediction\train.csv_VsW9EGx\train.csv")
test = pd.read_csv(r"E:\Hackathon\Janatahack Cross-sell Prediction\test.csv_yAFwdy2\test.csv")
pd.set_option("max_columns", 30)
pd.set_option("display.width", 2000)

# train.head(5)

le = LabelEncoder()
train["Gender"] = le.fit_transform(train["Gender"])
train["Vehicle_Age"] = le.fit_transform(train["Vehicle_Age"])
train["Vehicle_Damage"] = le.fit_transform(train["Vehicle_Damage"])

test["Gender"] = le.fit_transform(test["Gender"])
test["Vehicle_Age"] = le.fit_transform(test["Vehicle_Age"])
test["Vehicle_Damage"] = le.fit_transform(test["Vehicle_Damage"])

train['Region_Code'] = train['Region_Code'].astype(int)
test['Region_Code'] = test['Region_Code'].astype(int)
train['Policy_Sales_Channel'] = train['Policy_Sales_Channel'].astype(int)
test['Policy_Sales_Channel'] = test['Policy_Sales_Channel'].astype(int)

cat_col = ['Gender', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage','Policy_Sales_Channel']
# test.dtypes

f = plt.figure(figsize=(12, 10))
plt.matshow(train.corr(), fignum=f.number)
plt.xticks(range(train.shape[1]), train.columns, fontsize=11, rotation=90)
plt.yticks(range(train.shape[1]), train.columns, fontsize=11)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('', fontsize=16)
# plt.show()

train_x = train.drop(['Response', 'id'], axis=1)
train_y = train['Response']
test_x = test.drop(['id'], axis=1)
X_t, X_tt, y_t, y_tt = train_test_split(train_x, train_y, test_size=.25, random_state=150303, shuffle=True)

t1 = time.time()
clf = CatBoostClassifier(n_estimators=5000, loss_function='Logloss', eval_metric='LogLikelihoodOfPrediction', learning_rate=0.05, depth=9, boosting_type='Ordered', bagging_temperature=0.4, task_type='GPU', silent=True) # 0.8578
# clf = GaussianNB()
# clf = MLPClassifier(max_iter=300)
# clf = svm.SVC(kernel='linear')
t2 = time.time()
print(f"Model Created --- {(t2-t1) if (t2-t1)<60 else (t2-t1)/60}")


clf.fit(X_t, y_t, cat_features=cat_col, eval_set=(X_tt, y_tt), plot=True, early_stopping_rounds=30, verbose=100)
t3 = time.time()
print(f"Model Trained --- {(t3-t2)/60} Minutes")

sample = clf.predict_proba(test_x)

# print(sample[:, 1])

submit = pd.DataFrame(({"id": test.id, "Response": sample[:, 1]}))
submit.to_csv('submission.csv', index=False)
t4 = time.time()
print(f"Process Finished ---{(t4-t3)}sec")
