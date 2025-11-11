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
