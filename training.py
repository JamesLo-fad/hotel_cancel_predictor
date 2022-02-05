import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import pickle

df = pd.read_csv("https://drive.google.com/u/1/uc?id=17a2cEAPpkQc7hXwnmBr4_6xSAbSmNsMp&export=download")


def generate_datetime(row):
    month = datetime.strptime(row['arrival_date_month'], "%B").month

    return datetime(row['arrival_date_year'], month, row['arrival_date_day_of_month'])


df['dt'] = df[["arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"]].apply(generate_datetime, axis=1)
df1 = df.copy()
df1 = df1[[c for c in df1 if c not in ['is_canceled']] + ['is_canceled']]
df["number_of_occupant"] = df["adults"] + df["children"] + df["babies"]
df["has_baby"] = df["babies"] > 0
df["is_summer"] = df["arrival_date_month"].isin(["June", "July", "August"])

enc = OneHotEncoder()
one_hot_features = pd.DataFrame(enc.fit_transform(df[["hotel"]]).toarray(), columns=enc.get_feature_names())

x = df[["is_summer", "number_of_occupant"]].fillna(0)

x = pd.concat([x, one_hot_features], axis=1)
y = df["is_canceled"]

RFC = RandomForestClassifier()

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.30, random_state=42)

RFC.fit(X_train, y_train)

y_pred = RFC.predict(X_test)

with open('api_app/exported_one_hot.pickle', 'wb') as fp:
    pickle.dump(enc, fp)

with open('api_app/exported_classifier.pickle', 'wb') as fp:
    pickle.dump(RFC, fp)

