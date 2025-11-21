# Machine Learning Term Project

## 1. **Introduction**

본 프로젝트는 다양한 재료를 조합해 주문하는 샌드위치 체인점을 대상으로, 고객의 선호도 기반 맞춤형 샌드위치 조합 추천 시스템을 구축하는 것을 목표로 한다. 개인화된 추천 기능을 통해 고객 경험 및 만족도를 향상시키는 동시에, 재료 소비 패턴을 예측하여 신선도 관리 및 재고 관리 효율화 역시 도모하고자 한다.

### **1.1 문제 정의**

현재 대부분의 샌드위치 체인점은 사용자 취향을 반영한 **개인화 재료 조합 추천 시스템이 부재**하여 고객의 선택 피로도가 증가하고, 체계적인 재료 수요 예측이 어렵다는 문제가 발생한다.

### **1.2 프로젝트 목표**

1. 사용자별 재료 선호도를 분석해 개인화된 샌드위치 조합 추천 제공
2. 건강정보, 성별, 나이대 등 사용자 특성 정보를 활용한 맞춤 추천 기능 설계
3. 프로젝트를 위한 데이터셋 제작 → 통계적 관계 분석 → 모델의 수학적 원리 이해 및 적용

### **1.3 워크플로우**

이상적인 데이터 제작 후 훈련용 데이터셋 분리 → 모델링 (훈련용 데이터셋) → 평가 (이상적인 테스트용 데이터셋) → 결론 도출

---

## 2. **Data Description**

본 프로젝트는 목표에 적합한 공개 데이터셋이 없어, 직접 데이터를 제작하여 추천 시스템을 개발하였다. 생성된 데이터셋은 총 4종류로 구성된다.

### 2.1 사용자 데이터셋 (**user_info.csv)**

- 1000명의 사용자 정보 (1000×5)
- 성별, 연령대, 식습관, 채식 여부, 알레르기 정보를 포함

| **user_id** | **gender** | **age_group** | **diet** | **vegetarian** | **allergy** |
| --- | --- | --- | --- | --- | --- |
| user_001 | 0 | 0 | False | False | [] |
| user_002 | 1 | 0 | False | False | [] |
| user_003 | 1 | 3 | True | False | [] |
| user_004 | 1 | 0 | True | True | ['Meat_5', 'Vegtable_3'] |

### 2.2 샌드위치 재료 데이터셋 (**ingredient_nutrition.csv)**

- 20개의 재료 (4개 카테고리 × 5개 세부 재료)
- 재료별 칼로리 정보 포함
- 빵(Bread), 채소(Vegetable), 고기(Meat), 소스(Source)

| **Category** | **Ingredient** | **Calories** |
| --- | --- | --- |
| Bread_1 | 화이트 | 195.0 |
| … |  |  |
| Vegtable_1 | 양상추 | 2.9 |
| … |  |  |
| Meat_1 | 콩고기 | 150.0 |
| … |  |  |
| Source_1 | 스위트 어니언 | 40.1 |

### 2.3 샌드위치 조합 데이터셋 (combo.csv)

- 625개의 조합 (4개 카테고리에서 각 1개의 재료 선택)
- 원-핫 인코딩 적용 (625×20)

| **combo_id** | **Bread_1** | … | **Vegtable_1** | … | **Meat_5** | … | **Source_5** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C001 | 1 |  | 1 |  | 0 |  | 0 |
| C002 | 1 |  | 1 |  | 0 |  | 0 |
| C003 | 1 |  | 1 |  | 0 |  | 1 |
| C004 | 1 |  | 1 |  | 0 |  | 0 |

### 2.4 유저 평점 데이터셋 (**rating_dataset.csv)**

- 1000명 × 625조합 (희소행렬)
- 각 유저당 약 180개 조합에 대한 평점 존재, 나머지는 결측값

| user_id | C001 | … | C625 |
| --- | --- | --- | --- |
| user_001 | 3.36 |  | N/A |
| user_002 | N/A |  | 4.24 |
| user_003 | 2.38 |  | 4.6 |
| user_004 | 3.93 |  | 3.76 |

### 2.5 데이터 생성 플로우

1. 사용자별 20개의 재료에 대한 기본 선호도 생성 (랜덤, 0~5점, 0.5점 단위)
2. 재료별 다양성을 반영한 선호도 조정
    - 빵: 점수 편차 작음, 고기: 점수 편차 큼
3. 사용자 특성 반영: 채식 여부, 알레르기 정보 적용
4. 성별 및 연령대별 선호도 편차 적용
5. 20개 재료 선호도 × 4개 카테고리 비중 → 샌드위치 조합 평점 계산
6. 전체 평점 데이터셋 (1000명 × 625조합) 생성 → 테스트 데이터

### 2.6 모델 훈련 및 테스트 데이터셋 분리

- 인기 조합 기준 분류:
    - Top 100 조합
    - Second 200 조합
- Top 100 중 유저가 경험한 80개 샌드위치 평점 선택
- Second 200 중 유저가 경험한 100개 샌드위치 평점 선택
- 최종 트레인 데이터셋: 각 유저 180개 조합 평점 (현실적인 데이터셋 생성)

---

## 3. **Problem Definition & Strategy**

### 3.1 User-Based Collaborative Filtering

- 사용자의 평점 행렬에서 유사한 취향을 가진 다른 사용자를 찾아, 그들이 높게 평가한 아이템을 추천
- 유사도는 Pearson 상관계수로 계산하며, mean-centering으로 개인 편향을 보정

### 3.2 Item-Based Collaborative Filtering

- 아이템 간 평점 패턴의 유사도를 기반으로, 사용자가 과거에 좋아했던 아이템과 유사한 아이템을 추천
- Adjusted Cosine 유사도로 사용자 평균 평점을 보정하고, 상위 k개의 유사 아이템만 활용

### 3.3 Matrix Factorization (ALS)

- 사용자와 아이템을 latent factor 벡터로 표현하여 평점 행렬을 저차원으로 근사
- ALS(Alternating Least Squares) 알고리즘을 이용해 사용자 벡터와 아이템 벡터를 번갈아 학습, 전체 평균 평점을 기준으로 예측

### 3.4 하이브리드 추천 (User + Item + MF)

- UBCF, IBCF, MF 세 모델의 예측 점수를 **정규화 후 가중합**하여 최종 추천 점수를 생성합니다.
- 각 모델의 장점을 결합하여 정확도를 높이고, 제약 조건과 칼로리 우선순위 고려

---

## 4. **System Design**

### **4.1 Data Pipeline**

- CSV 로드
- 인덱스 설정 (`user_id`, `combo_id`)
- 기본 통계 계산 (`user_mean`, `item_mean`)

### **4.2 Preprocessing Module**

- **칼로리 계산 :** 각 조합별 칼로리 계산 (`combo_calories = df_combo.dot(cal_map)`)
- **채식/소이 필터링 :** 콩고기(SOY_ID) 감지, 단독 조합(SOY_ONLY_IDS) 선정
- **평점 행렬 재정렬 :** 모든 `combo_id` 기준 열 구성
- **제약 조건 처리 함수 :** 알레르기 필터링, 채식 필터링, 후보 조합 마스크 생성

### **4.3 Model Training Pipeline**

- Step 1: User-Based CF
    - Pearson 유사도 기반, Top-k 유사 사용자 선택
    - Mean-centering 예측 → `df_pred_user`
- Step 2: Item-Based CF
    - Adjusted Cosine 유사도 기반, Top-k 유사 아이템 선택
    - Weighted sum 예측 → `df_pred_item`
- Step 3: Matrix Factorization (ALS)
    - 사용자/아이템 잠재 요인(P, Q) 학습, ALS 반복 학습
    - 전체 평균 평점 기준 예측 → MF 모델 (`mf_model`)
- Step 4: 하이브리드 추천
    - UBCF + IBCF + MF 점수 정규화 후 가중합
    - 최종 점수 행렬 → `df_pred_hybrid`

### **4.4 추천 & 후처리**

- 후보 아이템 선정: 미평가 + 알레르기/채식 필터 적용
- Diet flag 고려: 점수 + 칼로리 가중합 또는 lexicographic 정렬
- 최종 상위 k 추천 산출 (예: 5개)
- Pretty print: 조합 ID, 점수, 칼로리, 재료 목록 표시

---

## 5. **Model Development**

User-Based CF, Item-Based CF, Matrix Factorization, 그리고 세 모델의 앙상블 방식으로 구성된 **하이브리드 추천 시스템의 모델링 구현 과정**을 설명함.  
구현은 모두 Python(Pandas + NumPy) 기반으로 이루어지며, 평점 행렬의 희소성뿐 아니라 사용자 제약 조건(알레르기·채식·다이어트)을 함께 처리하도록 코드 레벨에서 설계함. :contentReference[oaicite:0]{index=0}

---

### **5.1 User-Based Collaborative Filtering (UBCF)**

UBCF는 `predict_user_based(df, k=30)` 함수로 구현함.

- **입력/출력 구조**
  - 입력: `df`는 `user_id`를 인덱스로, `combo_id`를 컬럼으로 가지는 평점 DataFrame 사용.
  - 출력: 동일한 shape의 DataFrame을 반환하며, `NaN`이었던 위치를 예측 평점으로 채움.

- **유사도 계산 부분**
  - `pearson_sim(u, v)` 함수 사용.
  - 각 사용자 `u`에 대해 `df.loc[u]`를 기준으로, 다른 모든 사용자 `v`의 시리즈와 공통 평가 아이템만 골라 Pearson 상관계수 계산함.
  - 유사도는 딕셔너리 `user_sims[u] = [(v, sim_uv), ...]` 형태로 저장하며,  
    `sorted(..., key=lambda x: -abs(x[1]))[:k]` 방식으로 상위 k=30명의 이웃만 남김.
  - 유사도 0이거나 공통 평가가 거의 없는 경우는 `user_sims`에 포함하지 않음.

- **예측 부분**
  - 최종 DataFrame은 `pred = df.copy()`로 원본을 복사한 후, `NaN`인 위치만 덮어씀.
  - 각 사용자 `u`, 아이템 `i`에 대해:
    - 이미 평점이 있으면 그대로 사용.
    - `NaN`이면:
      - `user_sims[u]`에서 `(v, sim_uv)`를 순회하면서, `df.loc[v, i]`가 존재하는 이웃만 사용.
      - 분자: `sim_uv * (r_vi - mean_v)`의 합.
      - 분모: `|sim_uv|`의 합.
      - 예측값: `mean_u + (분자 / 분모)` 형태로 계산함.  
      - 분모가 0인 경우, 이웃이 없는 상황이므로 `mean_u`만 사용.

이 함수는 **Pandas의 행/열 접근과 리스트 기반 캐시(`user_sims`)를 활용하여**, 반복적인 유사도 재계산을 피하고, Top-k 이웃만 사용하는 구조로 구현함.

---

### **5.2 Item-Based Collaborative Filtering (IBCF)**

IBCF는 이웃 계산과 예측이 분리된 구조로 구현함.

#### **(1) 아이템 이웃 계산: `build_item_neighbors(df, k=20)`**

- **입력/출력**
  - 입력: 사용자×아이템 평점 DataFrame `df`.
  - 출력: `neighbors` 딕셔너리  
    → `neighbors[item_i] = [(item_j, sim_ij), ...]` 형태로 저장함.

- **유사도 계산**
  - 각 아이템 컬럼 이름 리스트를 `items = df.columns.tolist()`로 가져옴.
  - 이중 for문으로 `(i, j)` 쌍을 순회하며, `adjusted_cosine_item_sim(df[i], df[j], user_means)` 호출.
  - `user_means = df.mean(axis=1)`을 미리 계산해 사용자 평균을 저장해둔 후,  
    각 아이템 평점에서 해당 사용자의 평균을 빼서 중심화한 뒤 코사인 유사도 계산함.
  - 유사도가 0이 아닌 경우만 `sims.append((j, sim_ij))`에 추가함.
  - 최종적으로 `sims`를 절대값 기준으로 정렬해 상위 k=20개만 남기고 `neighbors[i]`에 저장함.

#### **(2) 평점 예측: `predict_item_based(df, neighbors)`**

- **기본 구조**
  - `pred = df.copy()`로 원본을 복사하고, `NaN` 위치만 예측값으로 채움.
  - `df.index`(user_id), `df.columns`(combo_id)를 그대로 유지하여, UBCF 결과와 같은 구조 사용.

- **예측 로직**
  - 사용자 `u`와 아이템 `i`에 대해:
    - 기존 평점이 있으면 그대로 사용.
    - `NaN`이면:
      - `neighbors[i]` 리스트를 순회하며 `(j, sim_ij)`를 가져옴.
      - `df.loc[u, j]`가 존재하는 경우에만:
        - 분자: `sim_ij * r_uj` 합산.
        - 분모: `|sim_ij|` 합산.
      - 분모가 0이 아니면 `num/den`을 예측값으로 사용.
      - 분모가 0이면 `item_mean[i]` (해당 아이템 전체 평균) 또는 전체 평균으로 교체함.

이 구조는 **이웃 정보(`neighbors`)를 한번만 계산해 캐싱**하고, 예측 시에는 해당 딕셔너리와 기존 평점 DataFrame만 참조하여 효율적으로 Item-based CF를 수행하도록 구현함.

---

### **5.3 Matrix Factorization (MF, ALS 기반)**

MF는 `MF` 클래스로 구현함. 클래스는 `fit(df)`와 `predict_user(user_id)` 두 메인 메서드를 제공함.

#### **5.3.1 학습: `fit(df)`**

- **데이터 변환**
  - `df.index`와 `df.columns`를 기반으로 사용자/아이템 ID를 정수 인덱스로 매핑함.
    - `uid2i`, `i2uid`, `iid2j`, `j2iid` 딕셔너리 사용.
  - `df`를 순회하며 `NaN`이 아닌 `(user_idx, item_idx, rating)` 튜플을 `rows` 리스트에 저장하고,  
    이를 NumPy 배열 `data`로 변환함.

- **파라미터 및 초기화**
  - `factors`(잠재 차원), `epochs`, `reg`(정규화 계수)는 생성자 인자로 설정함.
  - 사용자 수 `U`, 아이템 수 `I`, 잠재 차원 `K = self.factors`로 두고:
    - `P`는 shape `(U, K)`의 랜덤 초기화 행렬.
    - `Q`는 shape `(I, K)`의 랜덤 초기화 행렬.
  - 전체 평균 평점 `mu = data[:,2].mean()`을 사용함.

- **보조 구조**
  - `user_items[u] = [(i, r_ui), ...]`  
  - `item_users[i] = [(u, r_ui), ...]`  
  두 개의 딕셔너리에 사용자별/아이템별 관측 평점을 미리 저장해둠.

- **ALS 루프**
  - for epoch in range(epochs):  
    - **사용자 업데이트**  
      - 각 사용자 `u`에 대해:
        - 해당 사용자가 평가한 아이템들의 잠재 벡터를 모아 `Q_u` 생성.
        - 평점 벡터 `rs`에서 전체 평균 `mu`를 빼고,  
          `A = Q_u.T @ Q_u + reg * I`, `b = Q_u.T @ (rs - mu)`를 만든 후  
          `P[u] = np.linalg.solve(A, b)`로 업데이트함.
    - **아이템 업데이트**  
      - 각 아이템 `i`에 대해서도 동일 방식으로 `P_i`와 `rs`를 사용하여 `Q[i]`를 업데이트함.

#### **5.3.2 예측: `predict_user(user_id)`**

- 입력: 문자열 형태의 `user_id`.
- 처리:
  - 미리 저장한 `uid2i`에서 내부 인덱스 `ui`를 찾고,  
    `scores = mu + P[ui] @ Q.T` 연산으로 모든 아이템에 대한 예측값 벡터를 계산함.
  - `np.clip(scores, 0, 5)`로 평점 범위를 제한함.
  - 결과는 `pd.Series(scores, index=df.columns)` 형태로 반환함.

MF는 이처럼 **NumPy 행렬 연산과 `np.linalg.solve`를 직접 사용하여** ALS를 구현하였고,  
클래스 내부에 사용자/아이템 매핑과 잠재 행렬을 보관함으로써 이후 재사용이 가능하도록 설계함.

---

### **5.4 Hybrid Model (세 모델 가중 앙상블)**

하이브리드 모델은 `hybrid_predict(df_user, df_item, mf_model, ...)` 함수로 구현함.

- **입력**
  - `df_user` : UBCF가 예측한 평점 행렬.
  - `df_item` : IBCF가 예측한 평점 행렬.
  - `mf_model` : 위에서 학습한 `MF` 객체.
  - `w_user`, `w_item`, `w_mf` : 세 모델 가중치 (기본값 0.33, 0.33, 0.34).

- **MF 예측 행렬 생성**
  - 함수 내부에서 `for user_id in df_user.index:` 루프를 돌며  
    `mf_model.predict_user(user_id)` 호출.
  - 각 결과를 모아 `mf_scores` DataFrame 생성 (index/columns를 다른 둘과 동일하게 맞춤).

- **정규화 함수 `norm(df)`**
  - `lo = df.min().min()`, `hi = df.max().max()`로 전체 최소/최대값 계산.
  - `hi > lo`인 경우: `(df - lo) / (hi - lo + 1e-8)` 적용.
  - 값이 모두 같은 경우(hi == lo)는 0 또는 상수 행렬로 처리해 0 division 방지함.

- **가중합**
  - `U = norm(df_user)`, `I = norm(df_item)`, `M = norm(mf_scores)`로 정규화한 후,
  - `hybrid = w_user * U + w_item * I + w_mf * M` 계산.
  - 가중치는 함수 상단에서 `assert abs(w_user + w_item + w_mf - 1.0) < 1e-6`로 검증함.

최종적으로 이 함수는 **세 모델의 결과를 같은 스케일로 맞춰 가중합한 DataFrame**을 반환하고,  
이 결과가 이후 추천 단계에서 “기본 점수 행렬”로 사용됨.

---

### **5.5 사용자 제약 조건 반영 코드**

사용자 제약 조건은 독립적인 헬퍼 함수들로 구현함.

#### **(1) 알레르기 파싱: `_parse_allergy(val)`**

- `user_info.csv`의 `allergy` 컬럼 값이
  - `[]`, `['Meat_1', 'Vegtable_3']` 등 문자열 형태로 저장될 수 있음.
- `ast.literal_eval`을 사용해 실제 Python 리스트로 변환함.
- 에러 발생 시 빈 리스트 `[]`를 반환해 코드 안전성을 확보함.

#### **(2) 조합 필터링: `combos_with_restriction_mask(combo_df, restricted_ingredients)`**

- 입력: `combo_df` (0/1 원-핫 조합 행렬), 제한 재료 ID 리스트.
- 처리:
  - `restricted_ingredients` 중 `combo_df.columns`에 실제 존재하는 열만 필터링해 사용.
  - 해당 열들의 값을 행 단위로 합산하여, 합이 0보다 크면 해당 조합에 제한 재료가 포함된 것으로 간주함.
- 반환: `True`/`False` 시리즈로, True인 조합은 추천에서 제외함.

#### **(3) 채식 & 콩고기 메타 정보**

- 코드 상단에서:
  - `MEAT_IDS` : `"Meat_"` prefix를 가진 컬럼 목록.
  - `SOY_ID` : `ingredient_nutrition.csv`에서 “콩고기/soy”를 포함한 재료의 Category.
  - `SOY_ONLY_IDS` :  
    - 각 조합에서 `MEAT_IDS` 합이 1이고, 그 유일한 미트가 `SOY_ID`인 combo_id만 모아 리스트로 저장.
- 채식 사용자의 경우:
  - 추천 후보에서 `MEAT_IDS` 중 `SOY_ID`를 제외한 모든 미트 재료가 포함된 조합을 제거하고,
  - 추가로 `SOY_ONLY_IDS`에 속하는 조합만 남기도록 필터링함.

---

### **5.6 엔드투엔드 추천 파이프라인: `recommend_for_user`**

최종 추천은 `recommend_for_user(user_id, hybrid_scores, ...)` 함수에서 수행함.

- **입력**
  - `user_id` : 추천 대상 사용자.
  - `hybrid_scores` : 5.4에서 만든 하이브리드 점수 DataFrame.
  - `candidate_pool_n` : 상위 N개 후보 풀 크기 (기본 100).
  - `final_k` : 최종 추천 개수 (기본 5).
  - `diet_override`, `diet_rank_mode`, `w_score`, `w_cal` : 다이어트 모드 관련 파라미터.

- **주요 단계**
  1. `df_users.loc[user_id]`에서 `diet`, `vegetarian`, `allergy` 정보를 불러옴 (`_parse_allergy` 사용).
  2. `df_rating.loc[user_id].isna()`를 이용해, 아직 평가하지 않은 조합만 필터링함.
  3. 위 조합들에 대해 `hybrid_scores.loc[user_id]`를 기준으로 점수 내림차순 정렬하여 상위 후보를 구성함.
  4. 알레르기/채식 정보를 바탕으로 `combos_with_restriction_mask` 및 `SOY_ONLY_IDS`를 적용해  
     허용되지 않는 조합을 제거함.
  5. 남은 조합에 대해 칼로리(`combo_calories`)를 붙여 `pool_df`를 구성함.  
  6. `diet` 여부(또는 `diet_override`)에 따라:
     - 다이어트가 아니면 `score` 기준으로만 정렬.
     - 다이어트 모드면:
       - **`diet_rank_mode="weighted"`**:  
         점수와 칼로리에 대해 Z-score 정규화 후  
         `utility = 0.7 * z_score + 0.3 * (-z_calories)` 계산, `utility` 기준 정렬.
       - **`diet_rank_mode="lexi"`**:  
         `score` 내림차순, 동률 시 `calories` 오름차순 정렬.
  7. 최종 상위 `final_k`개 조합과 사용자 제약 정보를 딕셔너리 형태로 반환함.

- **출력 보조**
  - `ingredient_ids_for(combo_id)` : 해당 조합에 포함된 재료 ID 리스트 반환.
  - `format_combo_line(combo_id, rec_row, diet_flag)` :  
    combo_id, score, calories, (utility), ingredients를 한 줄 문자열로 포맷팅하여 콘솔에 출력하는 데 사용함.

이와 같이, 코드 전체는 **UBCF / IBCF / MF 모델의 예측 행렬 생성 → 하이브리드 점수 계산 → 제약 조건 필터링 → Top-k 추천 산출**의 흐름으로 구성되며,  
각 단계에서 Pandas DataFrame, 딕셔너리 캐시, NumPy 행렬 연산을 적절히 결합하여 구현함.

---

## 6. **Experiments & Evaluation**

---

## 7. **Result Analysis**

---

## 8. **Limitations & Future Work**

### **8.1 데이터 관련 한계**

**8.1.1 생성 데이터 기반 / 데이터 현실성 부족**

- 유저, 조합, 평점, 영양 데이터 모두 시뮬레이션 또는 임의 생성
- 실제 사용자 행동, 선호 패턴, 알레르기/채식 정보 등 현실과 다름
- 평점 행렬, 조합 구성, 콩고 단독 조합 등 현실 데이터와 차이 큼

**8.1.2 재료 간 상호작용 무시**

- 모든 계산이 선형 + 단순 유사도 → 현실적 추천 품질 판단 불가
- 재료 간 맛, 조화, 조리 방식 등 비선형적 상호작용 반영 불가

### **8.2 모델 기반 한계**

**8.2.1 User-Based / Item-Based CF**

- 평점이 생성 데이터라서 실제 유저 유사도를 정확히 반영 불가
- Pearson / Adjusted Cosine도 실제 패턴 없는 데이터에서는 의미 제한

**8.2.2 Matrix Factorization**

- 단순 $μ + P·Q$ 형태 : 생성 데이터에서 오버핏 위험성

**8.2.3 세 모델(UBCF, IBCF, MF) 점수 단순 합산**

- 선형 가중합 : 실제 재료 선호, 조합 선호, 맛/영양 상관관계 반영 불가

**8.2.4 정규화(min-max) 및 가중치 합산** 

- 생성 데이터 특성에 의존적, 현실성 반영 제한

### **8.3 성능 및 확장성 한계**

- 생성 데이터 기반이라 현실 데이터 미반영
- 모든 계산이 선형 + 단순 유사도 → 현실적 추천 품질 판단 불가
- 실시간 추천, cold-start 문제 등은 실제 데이터에서 검증 필요

### 8.4 향후 과제

**8.4.1 실제 데이터 적용**

- 현재 시스템은 **단순 선형 관계를 가정한 시뮬레이션 데이터**로 테스트
- 실제 사용자 평점, 구매 로그, 영양/알레르기 데이터 등 **현실 데이터를 수집**하여 모델 정확도 검증

**8.4.2 비선형 관계 및 복잡한 상호작용 반영**

- 현재 시스템은 선형 가중합 기반으로 동작, 복잡한 사용자-아이템 상호작용 반영 불가능
- 딥러닝 기반 추천 모델 등 비선형 모델 적용

**8.4.3 개인화 및 상황별 추천 강화**

- 현재는 채식, 알레르기 등 제약 조건만 일부 반영
- 실시간 행동 데이터, 컨텍스트를 반영한 개인화 추천 확장 필요

**8.4.4 하이브리드 모델 확장**

- 기존 가중합 방식 외에, **비선형 가중치 학습, 앙상블 학습** 등을 통해 추천 정확도 향상

---

## 9. **Conclusion**

- 프로젝트 요약
- 주요 성과
- 배운 점 및 서비스 적용 가능성
- 실질적인 비즈니스 가치

---

## 11. **Appendix**

### 11.1 팀원 역할

| 송영우 202135546 | 아이디어 제안, 데이터셋 제적 |
| --- | --- |
| 현관 202135596 | 추천 시스템 모델링 |
| 황성민 202135599 | 모델링 테스트 및 평가 |
| 오예진 202234908 | 최종 정리 및 발표 |

### 11.2 프로젝트 일정

| 9 Week | 아이디어 선정, proposal 작성 |
| --- | --- |
| 10 Week | 데이터셋 제작, 추천 시스템 초기 구축 |
| 11 Week | 추천 모델링 구축, 모델 테스트 및 평가 |
| 12 Week | 프로젝트 최종 보고서 작성, 발표 |

### 11.3 Git 주소

[https://github.com/Machine-Learning-team10](https://github.com/Machine-Learning-team10)
