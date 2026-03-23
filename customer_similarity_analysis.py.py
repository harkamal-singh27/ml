import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score

df = pd.read_csv("AWCustomers.csv")

selected_columns = [
    'BirthDate',
    'Education',
    'Occupation',
    'Gender',
    'MaritalStatus',
    'HomeOwnerFlag',
    'NumberCarsOwned',
    'NumberChildrenAtHome',
    'TotalChildren',
    'YearlyIncome'
]

df_selected = df[selected_columns].copy()

df_selected['BirthDate'] = pd.to_datetime(df_selected['BirthDate'])
today = pd.Timestamp.today()

df_selected['Age'] = (today - df_selected['BirthDate']).dt.days // 365
df_selected.drop(columns=['BirthDate'], inplace=True)

df_selected.fillna(df_selected.mode().iloc[0], inplace=True)

numeric_cols = [
    'Age',
    'NumberCarsOwned',
    'NumberChildrenAtHome',
    'TotalChildren',
    'YearlyIncome'
]

scaler = StandardScaler()
df_selected[numeric_cols] = scaler.fit_transform(df_selected[numeric_cols])

df_selected['IncomeGroup'] = pd.qcut(
    df_selected['YearlyIncome'],
    q=4,
    labels=['Low', 'Medium', 'High', 'Very High']
)

df_encoded = pd.get_dummies(df_selected, drop_first=True)

obj1 = df_encoded.iloc[0].values
obj2 = df_encoded.iloc[1].values

simple_matching = np.mean(obj1 == obj2)

obj1_bin = (obj1 > 0).astype(int)
obj2_bin = (obj2 > 0).astype(int)

jaccard = jaccard_score(obj1_bin, obj2_bin)

cosine_sim = cosine_similarity([obj1], [obj2])[0][0]


print("Simple Matching:", simple_matching)
print("Jaccard Similarity:", jaccard)
print("Cosine Similarity:", cosine_sim)

corr = df_selected['YearlyIncome'].corr(df_selected['NumberCarsOwned'])

print("Correlation:", corr)
