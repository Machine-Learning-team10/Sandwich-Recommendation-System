# 🥪 Machine Learning Term Project — Team 10  
**Team Members:**  
송영우 (202135546), 현관 (202135596), 황성민 (202135599), 오예진 (202234908)

---

## 1️⃣ Objective of the System

**Goal:**  
개인 맞춤형 샌드위치 추천 시스템을 통해 고객 경험과 만족도를 향상시키고,  
동시에 재료 신선도와 재고 관리 효율을 최적화하는 것.

**Assumptions:**
- 모든 샌드위치의 가격은 동일하다.  
- 각 재료 카테고리(Bread, Vegetable, Meat, Sauce)별로 하나의 재료만 선택 가능하다.  
- 사용자의 재료 선호도와 샌드위치 조합 평점 간의 관계는 **선형(linear)** 관계로 가정한다.

---

## 2️⃣ Datasets to Use

### 데이터 생성 절차 요약

1. **Initialize User Preferences**  
   - 500명의 사용자 각각에게 20개 재료에 대해 0~5점(0.5 단위)으로 선호도 부여  
   - 5개 재료 카테고리별 가중치 초기화  

2. **Apply Variations**  
   - 사용자별 평균 선호도 편향 추가  
   - 랜덤 노이즈 삽입  
   - 재료 카테고리별 분산 적용 (예: Bread는 낮은 분산, Meat는 높은 분산)

3. **Select Representative Combinations**  
   - 총 625개 가능한 조합 중 대표 50개 조합 선정  
   - 각 사용자에게 30개 조합 할당  

4. **Include Long-Tail Combinations**  
   - 남은 575개 조합 중 10개씩 추가로 배정하여 **롱테일 선호** 반영  

5. **Compute Combination Ratings**  
   - 재료별 선호도의 선형결합(linear combination)으로 조합 평점 계산  

6. **Apply Demographic and Dietary Adjustments**  
   - 성별, 연령대별 bias 추가  
   - 채식주의자 또는 알레르기 재료 포함 시 평점 -1점 패널티 적용  

7. **Generate Training and Testing Sets**  
   - 위 과정을 두 번 반복하여 **학습용(train)** 과 **테스트용(test)** 데이터셋 생성  

---

### Dataset Composition

#### 🧑 User Dataset
- 기본 사용자 정보 (gender, age, dietary info)  
- 재료별 선호 점수  
- 카테고리 가중치  
- 사용자 평균 편향(bias)

#### 🥬 Ingredient Dataset
| Category | Ingredient | Calories |
|-----------|-------------|-----------|
| Bread | White / Wheat / Parmesan Oregano / Honey Oat / Flatbread | 195~237 |
| Vegetable | Lettuce / Tomato / Pickle / Onion / Avocado | 2.9~56.5 |
| Meat | Roasted Chicken / Ham / Meatball / Bacon / Pepperoni | 40~210 |
| Sauce | Sweet Onion / Sweet Chili / Smoke BBQ / Honey Mustard / Ranch | 32~116 |

#### 🥪 Sandwich Composition Dataset
- 625 combinations × 20 ingredients (원핫 인코딩)
- 각 조합별 포함 재료 표시 (0/1)

#### 📊 Final Training Table
- 총 **20,000 user–sandwich 평점 데이터**  
- Columns: `user_id`, `sandwich_id`, `rating`

---

## 3️⃣ Filtering Methods to Use

### 3.1 User-Based Collaborative Filtering
- 사용자 간 유사도를 계산하여 **유사 사용자들의 평점 평균으로 예측**  
- 예측 평점 테이블 생성 시 활용  
- 개인 취향 기반 개인화 추천 수행  

### 3.2 Item-Based Collaborative Filtering
- 샌드위치 조합 간 유사도를 분석  
- **비슷한 조합의 재료 패턴**을 이용해 새로운 후보 추천  
- 다양한 조합 구성을 제시하여 **추천 다양성 확보**

### 3.3 Rule-Based / Attribute-Based Filtering
- 사용자의 건강 정보 및 식단 정보를 고려하여 다음 규칙 적용:
  - 알레르기 재료가 포함된 조합은 제외  
  - 채식주의자는 Meat 대신 **Soy-only 조합**으로 제한  
  - 다이어트 사용자는 **낮은 칼로리 조합에 가중치 부여**

---

### Recommendation System Workflow

1. **Matrix Factorization (ALS 기반 MF)**
   - 평점 테이블과 아이템 테이블을 이용해 MF 모델로 예측 평점 테이블 생성  
   - 구성 요소:  
     | Symbol | Description |
     |:-------|:------------|
     | **U (500×20)** | 사용자 잠재 벡터 |
     | **V (20×20)** | 재료 잠재 특성 벡터 |
     | **S (625×20)** | 샌드위치 조합 구성 벡터 |
     | **C = S·V (625×20)** | 샌드위치 임베딩 |
     | **R̂ = U·Cᵀ (500×625)** | 예측 평점 행렬 |

2. **User-Based CF**  
   - 사용자 간 유사도를 계산하여 예측 평점 행렬 생성

3. **Hybrid Predicted Ratings**  
   - User-based CF와 Item-based CF 예측 결과를 **가중 평균**으로 결합  
   - 상위 50개 샌드위치 후보 선정  

4. **Filtering**
   - 사용자 건강 정보 기반 필터 적용 (알레르기/식이 제한 반영)  
   - 이미 시도한 조합은 제외  

5. **Final Recommendation**
   - 최종 3개 샌드위치 조합 추천  
   - Item-based CF를 통해 **다양한 재료 구성**을 보장  

---

## 4️⃣ Machine Learning Model to Use

### 4.1 ALS-Based Matrix Factorization (MF)
- 사용자–샌드위치 평점 행렬을 분해하여 예측 평점 행렬 생성  
- 사용자, 샌드위치, 재료의 잠재 벡터를 학습  
- 반복 학습(ALS, Alternating Least Squares)을 통해 손실 함수 최소화  
- 대규모 희소 행렬(sparse matrix)에 최적화되어 효율적인 성능 발휘  

**장점:**  
- 데이터 희소성 문제 해결  
- 사용자 및 아이템 잠재 특성 파악 가능  
- 추천 정확도 향상  

---

## 📈 Summary

| 구분 | 접근 방식 | 목적 | 비고 |
|------|------------|------|------|
| **User-Based CF** | Memory-based | 유사 사용자 패턴 활용 | 개인화 추천 |
| **Item-Based CF** | Memory-based | 유사 조합 기반 추천 | 다양성 확보 |
| **Rule-Based Filtering** | Heuristic | 알레르기/채식/다이어트 반영 | 안전성 강화 |
| **Matrix Factorization (ALS)** | Model-based | 잠재 요인 학습 | 정확도 향상 |
| **Hybrid Combination** | Weighted Integration | 세 접근법 통합 | 성능·안정성 향상 |

---

## 🧭 Expected Outcomes
- 사용자의 **건강 정보와 취향을 모두 반영한 추천 시스템** 구축  
- 채식·알레르기·다이어트 등 **실제 제약조건을 고려한 실용적 추천**  
- 협업필터링과 행렬분해의 장점을 결합한 **하이브리드 모델**로 높은 추천 품질 달성
