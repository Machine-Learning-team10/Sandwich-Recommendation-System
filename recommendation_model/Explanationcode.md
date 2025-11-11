# 🥪 Personalized Sandwich Recommendation System  
**Machine Learning Team 10 — Final Implementation Summary**

> 본 프로젝트는 **사용자 맞춤형 샌드위치 추천 시스템**으로,  
> 사용자의 식습관(채식·다이어트), 알레르기 정보, 재료 선호도를 반영하여  
> 개별화된 추천을 제공합니다.

---

## 📘 Project Overview

| 항목 | 설명 |
|------|------|
| **Domain** | 음식 추천 (샌드위치 조합) |
| **Objective** | User Preference + Health Condition 기반 추천 |
| **Dataset 구성** | `user_info.csv`, `ingredient_nutrition.csv`, `combo.csv`, `rating_dataset.csv` |
| **Filtering Methods** | User-based CF, Item-based CF, Rule-based Filtering |
| **Machine Learning Model** | Biased Matrix Factorization (MF) |
| **Hybrid Method** | (UserCF + ItemCF + MF) Weighted Combination |
| **Special Rules** | Vegetarian → Soy-only Combos, Allergy → Exclude Ingredient Combos |
| **Diet Mode** | 0.7 × normalized(score) + 0.3 × normalized(-calories) Utility Ranking |

---

## 1) 목표와 전제
- **목표:** 사용자의 선호·제약(채식, 알레르기, 다이어트)을 반영한 **개인 맞춤형 샌드위치 조합 추천**.
- **데이터 전제:**
  - `combo.csv`: 조합(샌드위치)별 **원핫 재료 벡터**.
  - `ingredient_nutrition.csv`: 재료별 **영양정보(칼로리 포함)**.
  - `user_info.csv`: 사용자별 **diet / vegetarian / allergy** 플래그 및 메타.
  - `rating_dataset.csv`: 사용자–조합 **평점 행렬**(비어 있는 칸은 미평가).

---

## ⚙️ System Architecture

```text
[ Data Loading ] → [ User-based CF ] → [ Item-based CF ]
        ↓                         ↘
  [ Nutrition/Allergy Info ] → [ MF (Matrix Factorization) ]
        ↓
   [ Hybrid Integration (User + Item + MF) ]
        ↓
   [ Rule Filtering (Allergy/Vegetarian) ]
        ↓
   [ Diet-aware Re-ranking ]
        ↓
   [ Final Top-N Recommendation Output ]
```
 ---

## 2) 전체 로직 개요 (파이프라인)
1. **데이터 로딩 및 정합성 맞춤**: `rating`의 열과 `combo`의 인덱스를 `combo_id` 기준으로 일치.
2. **영양 매핑**: 재료 칼로리를 조합에 합산하여 **조합 칼로리** 생성.
3. **기본 모델들 계산**  
   - User-based CF (유저-유저 유사도 기반 예측)  
   - Item-based CF (아이템-아이템 유사도 기반 예측)  
   - Biased MF (잠재요인 기반 예측)
4. **하이브리드 통합**: 세 모델의 점수를 정규화 후 가중합하여 **최종 스코어** 생성.
5. **규칙 필터링**:  
   - 알레르기 재료 포함 조합 제외  
   - 채식주의자는 **콩고기 단독 미트 조합(Soy-only)** 만 후보로 제한
6. **개인화 랭킹**: 사용자의 diet 플래그에 따라  
   - diet=True → **유틸리티(점수+저칼로리)** 기반 재랭킹  
   - diet=False → **점수 우선** 정렬
7. **Top-K 추천 출력**: 최종 상위 K개 조합.

---

## 3) Filtering method to use 

### 3.1 User-based Collaborative Filtering
- **핵심 아이디어:** 나와 **유사한 사용자**(선호 패턴이 비슷한 사용자)를 찾고, 그들의 평가를 가중 평균하여 **내가 아직 평가하지 않은 조합의 점수**를 예측.
- **특징:**  
  - 사용자 평균 중심화(Mean-centering)로 평점 편향 보정  
  - 피어슨 상관계수 기반 유사도  
  - 장점: 개인 취향에 민감, 설명가능성(“유사한 유저들이 좋아함”)이 높음  
  - 한계: 사용자 수가 많고 평점이 희소한 경우 품질 저하 가능

### 3.2 Item-based Collaborative Filtering
- **핵심 아이디어:** 내가 좋아한(또는 높은 점수를 준) 조합과 **유사한 조합**을 찾고, 그 유사도와 내 기존 평점을 곱해 **새 조합의 점수**를 예측.
- **특징:**  
  - 사용자 평균 보정 후의 **Adjusted Cosine 유사도** 사용  
  - 장점: 아이템 구조(재료 조합의 유사성)를 잘 반영, 확장 추천에 유리  
  - 한계: 완전히 새로운 조합(콜드 아이템)에 대해선 제한적

### 3.3 Rule / Attribute-based Filtering
- **알레르기 제외:** 사용자 알레르기 목록에 포함된 재료가 **원핫=1**인 조합은 **완전히 제외**.  
- **채식주의자(vegetarian):** 후보군을 **‘콩고기 단독 미트’ 조합**으로 **강제 제한**.  
  - “콩고기/soy” 키워드로 콩고기 재료를 탐지(SOY_ID)하고,  
  - 미트 원핫 합계가 1이면서 SOY_ID=1인 조합만 허용.
- **다이어트(diet):** 랭킹 단계에서 **칼로리 패널티**를 적용(아래 4.2 참조).
- **의미:** 모델 예측의 품질과 무관하게, **사용자 제약조건을 철저히 보장**하는 안전장치.

---

## 4) Machine Learning model to use 

### 4.1 Biased Matrix Factorization (MF)
- **모델 수식:**  
  \( \hat{r}_{ui} = \mu + b_u + b_i + P_u^\top Q_i \)  
  - \(\mu\): 전체 평균 평점  
  - \(b_u, b_i\): 사용자/아이템 바이어스  
  - \(P_u, Q_i\): 사용자/아이템 잠재요인 벡터
- **학습:** SGD(확률적 경사하강)로 RMSE 최소화 방향 최적화  
- **장점:** 희소 데이터에서 **일반화** 성능이 뛰어나고, User/Item CF가 놓칠 수 있는 **잠재 패턴**을 포착  
- **역할:** User/Item CF와 결합했을 때 **하이브리드의 안정성**과 **성능 상한**을 끌어올림

### 4.2 Diet-aware Ranking (Utility)
- **목적:** diet=True 사용자에게는 “점수는 높지만 **칼로리는 낮은**” 조합을 우선 제시  
- **유틸리티 정의:**  
  \( \text{utility} = 0.7 \cdot z(\text{score}) + 0.3 \cdot z(-\text{calories}) \)  
  - 점수는 클수록 유리, 칼로리는 **낮을수록 유리**(음수 부호)  
  - 정규화 \(z(\cdot)\) 로 스케일 차이를 보정  
- **적용:** 후보 Top-N에서 utility 내림차순으로 최종 Top-K 선별

---

## 5) Hybrid Integration (UserCF + ItemCF + MF)
- **아이디어:** 세 접근법의 **보완성**을 활용하여 하나의 최종 점수로 통합  
- **절차:**  
  1. User-CF, Item-CF, MF 각 예측 스코어를 **min-max 정규화**  
  2. 가중치(예: 0.33/0.33/0.34)로 **가중합**  
- **효과:**  
  - User-CF: 개인 취향 민감도 ↑  
  - Item-CF: 유사 조합 확장성 ↑  
  - MF: 일반화/희소성 대응 ↑  
  → **정확도·안정성·다양성**을 균형 있게 확보

---

## 6) 최종 추천 흐름 (정리)
1. **세 모델 점수 계산:** User-CF, Item-CF, MF  
2. **하이브리드 스코어 생성:** 정규화 후 가중합  
3. **미평가 후보만 선별:** 이미 평가한 조합 제외  
4. **규칙 필터링:** 알레르기 제외 → 채식이면 Soy-only로 후보 제한  
5. **개인화 랭킹:**  
   - diet=True → 유틸리티 기준 정렬  
   - diet=False → 하이브리드 점수 기준 정렬  
6. **Top-K 반환:** 최종 추천 목록

---

## 7) 설계 상 고려 사항
- **정확성 vs. 제약 준수:** 모델이 높은 점수를 주더라도 **알레르기/채식 조건을 반드시 우선** 적용.  
- **데이터 일관성:** `combo_id`와 평점 행렬의 열, 재료 컬럼명 일치가 필수.  
- **확장성:**  
  - 다양성(MMR) 재랭킹 추가 가능  
  - 콜드스타트 사용자에 대한 초기 선호 주입  
  - 설명가능성(이웃 기여도/잠재요인 중요도) 출력 강화

---

## 8) 요약 표

| 분류 | 사용 기법 | 목적 | 핵심 포인트 |
|---|---|---|---|
| Filtering | User-based CF | 유사 사용자 활용 | Pearson, mean-centering |
| Filtering | Item-based CF | 유사 조합 확장 | Adjusted cosine |
| Filtering | Rule-based | 제약 준수 | Allergy 제외, Vegetarian=Soy-only |
| ML Model | Biased MF | 일반화/희소성 대응 | μ + bu + bi + P·Q (SGD) |
| Ranking | Hybrid | 모델 결합 | 정규화 후 가중합 |
| Ranking | Diet-aware | 건강 지향 랭킹 | 0.7·z(score)+0.3·z(-cal) |

---

## 9) 결과 출력 (데모 유저5명)
<img width="681" height="647" alt="스크린샷 2025-11-12 062154" src="https://github.com/user-attachments/assets/7c4aeb9f-93d6-4212-ac62-0698c73eb65e" />


---


## 10) 기대 효과
- **개인화**(User-CF) × **구조적 유사성**(Item-CF) × **일반화**(MF) × **안전성**(Rule)  
- **채식/알레르기/다이어트** 같은 **현실적 제약**을 **모델 출력 이전/이후 단계에서 모두 반영**  
- 가벼운 파라미터 튜닝만으로도 다양한 운영 목표(정확도·건강·다양성)에 맞춘 **유연한 추천** 가능

---
