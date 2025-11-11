# 🥪 Machine Learning Term Project — Team 10  
**Team Members and Roles:**  
Song Young-woo (202135546) - Data generation, proposal writing, ppt creation  
Hyun Gwan (202135596) - Model evaluation  
Hwang Sung-min (202135599) - System modeling, GitHub documentation  
Oh Ye-jin (202234908) - Presentation  

---

## 1️⃣ Objective of the System

**Goal:**  
To enhance customer experience and satisfaction through a **personalized sandwich recommendation system**,  
while simultaneously optimizing ingredient freshness and inventory management efficiency.

**Assumptions:**
- All sandwiches have the same price.  
- Each ingredient category (Bread, Vegetable, Meat, Sauce) allows only **one ingredient selection**.  
- The relationship between user ingredient preference and sandwich rating is assumed to be **linear**.

---

## 2️⃣ Datasets to Use

### Overview of Data Generation Process

1. **Initialize User Preferences**  
   - 500 users are assigned preference scores (0–5, step 0.5) for 20 ingredients.  
   - Initialize weights for 5 ingredient categories.

2. **Apply Variations**  
   - Add user-specific preference biases.  
   - Insert random noise.  
   - Apply category-specific variance (e.g., Bread has low variance, Meat has high variance).

3. **Select Representative Combinations**  
   - From 625 possible combinations, select 50 representative sandwich combinations.  
   - Assign 30 combinations per user.

4. **Include Long-Tail Combinations**  
   - Add 10 combinations randomly selected from the remaining 575 combinations  
     to reflect **long-tail preference**.

5. **Compute Combination Ratings**  
   - Calculate sandwich ratings as a **linear combination** of ingredient preferences.

6. **Apply Demographic and Dietary Adjustments**  
   - Add bias based on gender and age group.  
   - Apply a -1 point penalty if a sandwich contains ingredients the user is allergic to  
     or violates vegetarian restrictions.

7. **Generate Training and Testing Sets**  
   - Repeat the above process twice to create **train** and **test** datasets.

---

### Dataset Composition

#### 🧑 User Dataset
- Basic user information (gender, age, dietary info)  
- Ingredient preference scores  
- Category weights  
- User average bias  

#### 🥬 Ingredient Dataset
| Category | Ingredient | Calories |
|-----------|-------------|-----------|
| Bread | White / Wheat / Parmesan Oregano / Honey Oat / Flatbread | 195–237 |
| Vegetable | Lettuce / Tomato / Pickle / Onion / Avocado | 2.9–56.5 |
| Meat | Roasted Chicken / Ham / Meatball / Bacon / Pepperoni | 40–210 |
| Sauce | Sweet Onion / Sweet Chili / Smoke BBQ / Honey Mustard / Ranch | 32–116 |

#### 🥪 Sandwich Composition Dataset
- 625 combinations × 20 ingredients (one-hot encoded)  
- Indicates inclusion of each ingredient (0/1)

#### 📊 Final Training Table
- Total of **20,000 user–sandwich rating data points**  
- Columns: `user_id`, `sandwich_id`, `rating`

---

## 3️⃣ Filtering Methods to Use

### 3.1 User-Based Collaborative Filtering
- Calculate similarity between users and **predict missing ratings** using weighted averages  
  from similar users.  
- Build predicted rating tables.  
- Provides **personalized recommendations** based on individual preferences.

### 3.2 Item-Based Collaborative Filtering
- Analyze similarity between sandwich combinations.  
- Recommend new candidates using **ingredient pattern similarity**.  
- Promotes **diversity of recommendations** through variation in combinations.

### 3.3 Rule-Based / Attribute-Based Filtering
- Apply rules based on users’ health and dietary information:
  - Exclude sandwiches containing allergenic ingredients.  
  - For vegetarians, limit to **soy-only meat combinations**.  
  - For diet users, apply **additional weight to low-calorie sandwiches**.

---

### Recommendation System Workflow

1. **Matrix Factorization (ALS-based MF)**  
   - Generate predicted rating matrix using MF on the user–sandwich matrix.  
   - Components:  

     | Symbol | Description |
     |:-------|:------------|
     | **U (500×20)** | User latent feature matrix |
     | **V (20×20)** | Ingredient latent feature matrix |
     | **S (625×20)** | Sandwich composition matrix |
     | **C = S·V (625×20)** | Sandwich embedding |
     | **R̂ = U·Cᵀ (500×625)** | Predicted rating matrix |

2. **User-Based CF**  
   - Compute user similarity and generate predicted ratings.

3. **Hybrid Predicted Ratings**  
   - Combine User-based and Item-based CF results using **weighted averaging**.  
   - Select top 50 sandwich candidates.

4. **Filtering**  
   - Apply filtering based on users’ health data (allergy/dietary restrictions).  
   - Exclude sandwiches the user has already tried.

5. **Final Recommendation**  
   - Recommend top 3 sandwiches.  
   - Use Item-based CF to ensure **diverse ingredient combinations**.

---

## 4️⃣ Machine Learning Model to Use

### 4.1 ALS-Based Matrix Factorization (MF)
- Decompose the user–sandwich rating matrix to predict missing ratings.  
- Learn latent vectors for users, sandwiches, and ingredients.  
- Use **Alternating Least Squares (ALS)** to minimize loss iteratively.  
- Optimized for large-scale **sparse matrices**.

**Advantages:**  
- Addresses data sparsity issues.  
- Captures latent relationships between users and items.  
- Improves overall recommendation accuracy.

---

## 📈 Summary

| Category | Approach | Objective | Remarks |
|-----------|-----------|-----------|----------|
| **User-Based CF** | Memory-based | Utilize similar user patterns | Personalized recommendation |
| **Item-Based CF** | Memory-based | Recommend similar combinations | Ensures diversity |
| **Rule-Based Filtering** | Heuristic | Apply allergy/vegetarian/diet restrictions | Increases safety |
| **Matrix Factorization (ALS)** | Model-based | Learn latent features | Improves accuracy |
| **Hybrid Combination** | Weighted Integration | Combine multiple methods | Enhances performance & stability |

---

## 🧭 Expected Outcomes
- A **personalized recommendation system** that considers both user preferences and health information.  
- Practical recommendation reflecting **realistic constraints** such as vegetarianism, allergies, and diet.  
- A **hybrid model** combining collaborative filtering and matrix factorization to achieve high-quality recommendations.
