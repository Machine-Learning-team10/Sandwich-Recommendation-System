import numpy as np
import pandas as pd

np.random.seed(42)

#0 변수 초기화
categories = ["Bread", "Vegtable", "Meat", "Source"]
ingredients_ids = [f"{cat}_{i+1}" for cat in categories for i in range(5)]
n_users = 1000
user_ids = [f"user_{i+1:03d}" for i in range(n_users)]
n_categories = 4
cat_size = 5 
ingredients_size = 20

p_female = 0.5
gender = np.random.choice([0,1], size=n_users, p=[1-p_female, p_female])
p_diet = 0.3
diet = np.random.choice([False, True], size=n_users, p=[1-p_diet, p_diet])
# 0 : 10대, 1 : 20~30대, 2: 40~50대, 3 : 60대 이상
age_groups = np.random.choice([0,1,2,3], size=n_users, p=[0.3,0.4,0.2,0.1])

df_users = pd.DataFrame({
    "gender": gender,
    "age_group": age_groups,
    "diet": diet
},index=user_ids)

data = {
    "Category": ingredients_ids,
    "Ingredient": ["화이트", "위트", "파마산 오레가노", "허니오트", "플랫브레드",
                   "양상추", "토마토", "피클", "양파", "아보카도",
                   "콩고기", "햄", "미트볼", "베이컨", "페퍼로니",
                   "스위트 어니언", "스위트 칠리", "스모크 바비큐", "허니 머스타드", "랜치"],
    "Calories": [195, 195, 207, 237,232,
                 2.9, 7.7, 0.4, 2.8, 56.5,
                 150, 40, 210, 90, 150,
                 40.1, 40, 32.8, 38.4, 116]
}

df_nutrition = pd.DataFrame(data)

#1 유저의 각 20개의 재료별 선호도 + 재료별 다양성 (빵은 비슷한 점수, 고기는 편차 큼) 적용
data = np.random.choice(np.arange(1, 6, 0.5), size=(n_users, 20))
sigma = { "Bread": 0.1, "Vegtable": 0.15, "Meat": 0.35, "Source": 0.25 }
noise = np.zeros_like(data)
for idx, (cat, s) in enumerate(sigma.items()):
    start = idx*5
    end = start+5
    noise[:, start:end] = np.random.normal(0, s, size=(n_users, 5))
data = np.clip(data + noise, 0, 5)
data_array = np.array(data)

df_data = pd.DataFrame(data_array, index=user_ids, columns=ingredients_ids)

#1-1 채식주의자 정보 적용
p_veg = 0.15 
veg_users = np.random.choice([True, False], size=n_users, p=[p_veg, 1-p_veg])
df_users["vegetarian"] = veg_users
data_array[veg_users, 11:15] = -1
data_array[veg_users, 10] = 5


#1-2  알레르기 정보 적용
allergy_ingredients_list = [[] for _ in range(n_users)]
p_allergy = 0.10  
n_selected = int(n_users * p_allergy)
selected_users = np.random.choice(n_users, size=n_selected, replace=False)
for selected_user in selected_users:
    n_zero = np.random.randint(1, 4) 
    cols = np.random.choice(ingredients_size, size=n_zero, replace=False)
    data_array[selected_user, cols] = -1
    allergy_ingredients_list[selected_user] = [ingredients_ids[i] for i in cols]

df_users["allergy"] = allergy_ingredients_list

# 성별 연령별 선호도 편차 적용
rating_offset = np.zeros_like(data_array, dtype=float)

for i in range(n_users):
    gender_i = df_users.loc[user_ids[i], "gender"]
    age_i = df_users.loc[user_ids[i], "age_group"]
    
    if age_i <= 1: # 20~30대 화이트, 허니오트 선호
        rating_offset[i, 0] += 1  # 화이트
        rating_offset[i, 1] -= 1  # 위트
        rating_offset[i, 2] -= 1  # 파마산
        rating_offset[i, 3] += 1  # 허니오트
    elif age_i >= 2: # 50대 위트, 파마산 선호
        rating_offset[i, 0] -= 1 # 화이트
        rating_offset[i, 1] += 1 # 위트
        rating_offset[i, 2] += 1 # 파마산
        rating_offset[i, 3] -= 1  # 허니오트
    
    # Vegetables
    if gender_i == 1: # 여성 토마토, 아보카도 선호
        rating_offset[i, 5] -= 1  # 양상추
        rating_offset[i, 6] += 1  # 토마토
        rating_offset[i, 7] -= 1  # 피클
        rating_offset[i, 9] += 1  # 아보카도
    else: # 남성 양상추, 피클 선호
        rating_offset[i, 5] += 1  # 양상추
        rating_offset[i, 6] -= 1  # 토마토
        rating_offset[i, 7] += 1  # 피클
        rating_offset[i, 9] -= 1  # 아보카도
    
    # Meat
    if gender_i == 0: # 남성 미트볼 베이컨 선호
        rating_offset[i, 11] -= 1  # 햄
        rating_offset[i, 12] += 1 # 미트볼
        rating_offset[i, 13] -= 1 # 베이컨
        rating_offset[i, 14] += 1 # 페퍼로니
    else: # 여성 햄, 베이컨 선호
        rating_offset[i, 11] += 1  # 햄
        rating_offset[i, 12] -= 1 # 미트볼
        rating_offset[i, 13] += 1 # 베이컨
        rating_offset[i, 14] -= 1 # 페퍼로니
    
    # Sauce
    if gender_i == 0: 
        rating_offset[i, 15] -= 1 # 스위트 어니언
        rating_offset[i, 16] -= 1 # 스위트 칠리
        rating_offset[i, 17] += 1 # 스모크 바비큐
        rating_offset[i, 19] += 1 # 랜치
    else:
        rating_offset[i, 15] += 1 # 스위트 어니언
        rating_offset[i, 16] += 1 # 스위트 칠리
        rating_offset[i, 17] -= 1 # 스모크 바비큐
        rating_offset[i, 19] -= 1 # 랜치

alpha = 0.2
data_array = np.where(data_array != -1, data_array + alpha*rating_offset, -1)

#1-4 유저의 20개의 재료별 선호도 x 4개 카테고리별 비중
user_categories_weight = np.random.dirichlet(np.ones(n_categories), size=n_users)
user_categories_weight_expanded = np.repeat(user_categories_weight, cat_size, axis=1)
weighted_data = data_array * user_categories_weight_expanded
weighted_array = np.array(weighted_data)

#2 전체 샌드위치 조합 데이터 (625조합x20재료)
combo_list = []
for b in range(5):
    for v in range(5):
        for m in range(5):
            for s in range(5):
                combo = [0]*20
                combo[b] = 1        
                combo[5 + v] = 1        
                combo[10 + m] = 1     
                combo[15 + s] = 1      
                combo_list.append(combo)
combo_array = np.array(combo_list)

df_combo = pd.DataFrame(combo_array, 
                        index= [f"C{i+1:03d}" for i in range(combo_array.shape[0])],
                        columns=ingredients_ids)

 
#2-1 유저의 전체 샌드위치 점수 - 선형조합
Rating = weighted_array @ combo_array.T
Rating = np.clip(Rating, 0, 5)
Rating = np.round(Rating, 2) 

df_rating = pd.DataFrame(Rating, 
                         index=user_ids,
                         columns=[f"C{i+1:03d}" for i in range(Rating.shape[1])])

#3 top100 인기 조합, second200 인기 조합 분류
combo_mean = df_rating.mean(axis=0)
sorted_combos = combo_mean.sort_values(ascending=False).index 
n_top = 100
top_cols_idx = sorted_combos[:n_top]
n_second = 200
second_cols_idx = sorted_combos[n_top:n_top+n_second]
df_top = df_rating[top_cols_idx]
df_rest = df_rating[second_cols_idx]


#3-1 top 100개중 유저들이 먹어본 80개 샌드위치 평점 선정
n_keep = 80
df_top_50 = df_top.copy()  
for i in range(df_top_50.shape[0]):  
    cols = df_top_50.columns
    keep_cols = np.random.choice(cols, size=n_keep, replace=False)  
    drop_cols = [c for c in cols if c not in keep_cols] 
    df_top_50.iloc[i, df_top_50.columns.get_indexer(drop_cols)] = np.nan


#3-1 second 200개중 유저들이 먹어본 100개 샌드위치 평점 선정
n_keep = 100
df_rests = df_rest.copy()  
for i in range(df_rests.shape[0]):  
    cols = df_rests.columns
    keep_cols = np.random.choice(cols, size=n_keep, replace=False) 
    drop_cols = [c for c in cols if c not in keep_cols]  
    df_rests.iloc[i, df_rests.columns.get_indexer(drop_cols)] = np.nan

#4 최종 데이터셋 제작
df_final = pd.concat([df_top_50, df_rests], axis=1)
df_final = df_final.reindex(sorted(df_final.columns), axis=1)
print(df_final) 


print(df_final[df_users["vegetarian"]])

#4-1 데이터셋 저장
df_rating.to_csv("test_rating_dataset.csv", index=True, index_label="user_id") #Test Dataset
df_final.to_csv("rating_dataset.csv", index=True, index_label="user_id") #Train Dataset
df_users.to_csv("user_info.csv", index=True, index_label="user_id")
df_nutrition.to_csv("ingredient_nutrition.csv", index=False)
df_combo.to_csv("combo.csv", index=True, index_label="combo_id")
