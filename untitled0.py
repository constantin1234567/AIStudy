# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:29:31 2025

@author: robot
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 16:07:03 2025

@author: robot
"""



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import  OrdinalEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer  # Imputer 추가

# 파일 경로 설정 (train.csv)
file_path_train = r'C:\Users\robot\OneDrive\Desktop\python\train.csv'  # 실제 파일 경로로 변경
# 데이터 불러오기
train = pd.read_csv(file_path_train).drop(columns=['ID'])

# 파일 경로 설정 (test.csv)
file_path_test = r'C:\Users\robot\OneDrive\Desktop\python\test.csv'
# 데이터 불러오기
test = pd.read_csv(file_path_test).drop(columns=['ID'])


# 데이터 확인
#print(df.head())

# 데이터 분할: X(특성)과 y(목표 변수)
X = train.drop('임신 성공 여부', axis=1)  # "성공 여부"를 제외한 특성
y = train["임신 성공 여부"]  # "성공 여부"는 목표 변수

# 범주형 변수를 처리하기 위한 컬럼 리스트
categorical_columns = [
    "시술 시기 코드",
    "시술 당시 나이",
    "시술 유형",
    "특정 시술 유형",
    "배란 자극 여부",
    "배란 유도 유형",
    "단일 배아 이식 여부",
    "착상 전 유전 검사 사용 여부",
    "착상 전 유전 진단 사용 여부",
    "남성 주 불임 원인",
    "남성 부 불임 원인",
    "여성 주 불임 원인",
    "여성 부 불임 원인",
    "부부 주 불임 원인",
    "부부 부 불임 원인",
    "불명확 불임 원인",
    "불임 원인 - 난관 질환",
    "불임 원인 - 남성 요인",
    "불임 원인 - 배란 장애",
    "불임 원인 - 여성 요인",
    "불임 원인 - 자궁경부 문제",
    "불임 원인 - 자궁내막증",
    "불임 원인 - 정자 농도",
    "불임 원인 - 정자 면역학적 요인",
    "불임 원인 - 정자 운동성",
    "불임 원인 - 정자 형태",
    "배아 생성 주요 이유",
    "총 시술 횟수",
    "클리닉 내 총 시술 횟수",
    "IVF 시술 횟수",
    "DI 시술 횟수",
    "총 임신 횟수",
    "IVF 임신 횟수",
    "DI 임신 횟수",
    "총 출산 횟수",
    "IVF 출산 횟수",
    "DI 출산 횟수",
    "난자 출처",
    "정자 출처",
    "난자 기증자 나이",
    "정자 기증자 나이",
    "동결 배아 사용 여부",
    "신선 배아 사용 여부",
    "기증 배아 사용 여부",
    "대리모 여부",
    "PGD 시술 여부",
    "PGS 시술 여부"
]

numeric_columns = [
    "임신 시도 또는 마지막 임신 경과 연수",
    "총 생성 배아 수",
    "미세주입된 난자 수",
    "미세주입에서 생성된 배아 수",
    "이식된 배아 수",
    "미세주입 배아 이식 수",
    "저장된 배아 수",
    "미세주입 후 저장된 배아 수",
    "해동된 배아 수",
    "해동 난자 수",
    "수집된 신선 난자 수",
    "저장된 신선 난자 수",
    "혼합된 난자 수",
    "파트너 정자와 혼합된 난자 수",
    "기증자 정자와 혼합된 난자 수",
    "난자 채취 경과일",
    "난자 해동 경과일",
    "난자 혼합 경과일",
    "배아 이식 경과일",
    "배아 해동 경과일"
]
# 범주형 변수의 NaN 값을 'Unknown'으로 채움
#df[categorical_columns] = df[categorical_columns].fillna('Unknown')

# 숫자형 변수 처리
#numeric_columns = X.select_dtypes(include=["float64", "int64"]).columns.tolist()

# 데이터 분할: 훈련 데이터와 테스트 데이터
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 카테고리형 컬럼들을 문자열로 변환
for col in categorical_columns:
    X[col] = X[col].astype(str)
    test[col] = test[col].astype(str)

ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

X_train_encoded = X.copy()
X_train_encoded[categorical_columns] = ordinal_encoder.fit_transform(X[categorical_columns])

X_test_encoded = test.copy()
X_test_encoded[categorical_columns] = ordinal_encoder.transform(test[categorical_columns])

X_train_encoded[numeric_columns] = X_train_encoded[numeric_columns].fillna(0)
X_test_encoded[numeric_columns] = X_test_encoded[numeric_columns].fillna(0)




# RandomForest 모델 학습을 위한 파이프라인
#model = Pipeline(steps=[    
#    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))  # RandomForest 분류기
#])

# 모델 학습
print("model fit before");
model = ExtraTreesClassifier(random_state=42)

model.fit(X_train_encoded, y)
print("model fit finished");
# 예측
#y_pred = model.predict(X_test)

# 평가 결과 출력
#print("✅ 모델 정확도:", accuracy_score(y_test, y_pred))
#print("📊 분류 보고서:\n", classification_report(y_test, y_pred))
