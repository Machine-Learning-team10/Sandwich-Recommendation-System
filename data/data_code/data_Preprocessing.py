import pandas as pd

#1 데이터 불러오기
df_rating= pd.read_csv("rating_dataset.csv",  index_col="user_id")
df_users= pd.read_csv("user_info.csv",  index_col="user_id")
df_nutrition=pd.read_csv("ingredient_nutrition.csv")
df_combo=pd.read_csv("combo.csv", index_col="combo_id")

#2 데이터 전처리 - 1000x625 데이터셋으로 확장
all_combo_cols = df_combo.index.tolist()
df_rating = df_rating.reindex(columns=all_combo_cols).astype("float64")

#2-1 데이터 전처리 - 유저별 평균 편차 제거 
user_mean = df_rating.mean(axis=1)
df_rating_demean = df_rating.sub(user_mean, axis=0)

print(df_rating_demean)

