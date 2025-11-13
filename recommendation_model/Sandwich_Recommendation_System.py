import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
import ast
import os


# 데이터 로드 (Google Colab 기준)
# from google.colab import drive
# drive.mount('/content/drive')

# 경로 설정(상대경로로 업데이트)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # recommendation_model/
ROOT_DIR = os.path.dirname(BASE_DIR)                    # Sandwich-Recommendation-System/
PATH = os.path.join(ROOT_DIR, "data")                   # data/

df_rating = pd.read_csv(os.path.join(PATH, "rating_dataset.csv"), index_col="user_id")
df_users  = pd.read_csv(os.path.join(PATH, "user_info.csv"),     index_col="user_id")
df_nutri  = pd.read_csv(os.path.join(PATH, "ingredient_nutrition.csv"))
df_combo  = pd.read_csv(os.path.join(PATH, "combo.csv"),         index_col="combo_id")


# 조합 칼로리 & 메타 (SOY 감지 + SOY_ONLY 목록)
cal_map = df_nutri.set_index("Category")["Calories"]
valid_cols = [c for c in df_combo.columns if c in cal_map.index]
combo_calories = (df_combo[valid_cols].dot(cal_map.loc[valid_cols])).rename("calories")

MEAT_IDS = [c for c in df_combo.columns if c.lower().startswith("meat_")]
# 콩고기(soy meat) 자동 감지
soy_mask = df_nutri.get("Ingredient", pd.Series(["" for _ in range(len(df_nutri))]))\
    .astype(str).str.contains("콩고기|soy", case=False, regex=True)
SOY_ID: Optional[str] = None
if soy_mask.any():
    SOY_ID = str(df_nutri.loc[soy_mask, "Category"].iloc[0])

# 채식 전용: 콩고기 '단독 미트' 조합만 (미트합==1 & SOY_ID==1)
SOY_ONLY_IDS: List[str] = []
if SOY_ID is not None and SOY_ID in df_combo.columns:
    meat_sum = df_combo[MEAT_IDS].sum(axis=1)
    soy_flag = df_combo[SOY_ID] == 1
    SOY_ONLY_IDS = df_combo.index[(meat_sum == 1) & (soy_flag)].tolist()

# 평점 테이블 정렬 (모든 combo_id 기준 열 구성)
all_combo_cols = df_combo.index.tolist()
df_rating = df_rating.reindex(columns=all_combo_cols)

# 통계
user_mean = df_rating.mean(axis=1)
item_mean = df_rating.mean(axis=0)


# 유사도 함수
def pearson_sim(u: pd.Series, v: pd.Series) -> float:
    both = u.notna() & v.notna()
    if both.sum() < 2:
        return 0.0
    val = u[both].corr(v[both])
    return float(val) if pd.notna(val) else 0.0


def adjusted_cosine_item_sim(i: pd.Series, j: pd.Series, user_means: pd.Series) -> float:
    both = i.notna() & j.notna()
    if both.sum() < 2:
        return 0.0
    xi = i[both] - user_means[both]
    xj = j[both] - user_means[both]
    num = float((xi * xj).sum())
    den = float(np.sqrt((xi**2).sum()) * np.sqrt((xj**2).sum()))
    return (num / den) if den != 0.0 else 0.0

# User-Based CF (Pearson, mean-centering)

def predict_user_based(df: pd.DataFrame, k:int=30) -> pd.DataFrame:
    users = df.index
    items = df.columns
    means = df.mean(axis=1)
    df_pred = df.copy()

    user_sims: Dict[str, List[Tuple[str, float]]] = {}
    for u in users:
        sims = []
        u_vec = df.loc[u]
        for v in users:
            if u == v:
                continue
            s = pearson_sim(u_vec, df.loc[v])
            if s != 0.0:
                sims.append((v, s))
        sims = sorted(sims, key=lambda x: abs(x[1]), reverse=True)[:k]
        user_sims[u] = sims

    for u in users:
        mu = means[u]
        for i in items:
            if not pd.isna(df.at[u, i]):
                continue
            num = 0.0
            den = 0.0
            for v, w in user_sims[u]:
                rv_i = df.at[v, i]
                if pd.isna(rv_i):
                    continue
                num += w * (rv_i - means[v])
                den += abs(w)
            df_pred.at[u, i] = mu + (num/den if den > 0 else 0.0)
    return df_pred


# 5) Item-Based CF (Adjusted Cosine)

def build_item_neighbors(df: pd.DataFrame, k:int=20) -> Dict[str, List[Tuple[str, float]]]:
    items = df.columns.tolist()
    neighbors: Dict[str, List[Tuple[str, float]]] = {}
    for idx_i, i in enumerate(items):
        sims = []
        si = df[i]
        for idx_j, j in enumerate(items):
            if idx_i == idx_j:
                continue
            sj = df[j]
            s = adjusted_cosine_item_sim(si, sj, user_mean)
            if s != 0.0:
                sims.append((j, s))
        sims = sorted(sims, key=lambda x: abs(x[1]), reverse=True)[:k]
        neighbors[i] = sims
    return neighbors


def predict_item_based(df: pd.DataFrame, neighbors: Dict[str, List[Tuple[str, float]]]) -> pd.DataFrame:
    users = df.index
    items = df.columns
    df_pred = df.copy()

    for u in users:
        for i in items:
            if not pd.isna(df.at[u, i]):
                continue
            num = 0.0
            den = 0.0
            for j, s in neighbors[i]:
                r_uj = df.at[u, j]
                if pd.isna(r_uj):
                    continue
                num += s * r_uj
                den += abs(s)
            if den > 0:
                df_pred.at[u, i] = num / den
            else:
                df_pred.at[u, i] = item_mean[i] if pd.notna(item_mean[i]) else float(df.values.mean())
    return df_pred


# Biased Matrix Factorization (MF, SGD)

class MF:
    def __init__(self, factors: int = 40, epochs: int = 15, lr: float = 0.01, reg: float = 0.02, seed: int = 42):
        self.factors, self.epochs, self.lr, self.reg, self.seed = factors, epochs, lr, reg, seed

    def fit(self, df: pd.DataFrame):
        rng = np.random.RandomState(self.seed)
        self.users = df.index.tolist()
        self.items = df.columns.tolist()
        uid2i = {u:i for i,u in enumerate(self.users)}
        iid2j = {v:j for j,v in enumerate(self.items)}

        rows = []
        for u in self.users:
            for i in self.items:
                r = df.at[u, i]
                if not pd.isna(r):
                    rows.append((uid2i[u], iid2j[i], float(r)))
        data = np.array(rows, dtype=np.int64)
        U, I = len(self.users), len(self.items)
        self.P = 0.1 * rng.randn(U, self.factors)
        self.Q = 0.1 * rng.randn(I, self.factors)
        self.bu = np.zeros(U)
        self.bi = np.zeros(I)
        self.mu = float(np.mean([r for _,_,r in rows])) if rows else 0.0

        for ep in range(self.epochs):
            rng.shuffle(data)
            for ui, ij, r in data:
                pred = self.mu + self.bu[ui] + self.bi[ij] + self.P[ui] @ self.Q[ij]
                err = r - pred
                # updates
                self.bu[ui] += self.lr * (err - self.reg*self.bu[ui])
                self.bi[ij] += self.lr * (err - self.reg*self.bi[ij])
                pu = self.P[ui].copy(); qi = self.Q[ij].copy()
                self.P[ui] += self.lr * (err*qi - self.reg*pu)
                self.Q[ij] += self.lr * (err*pu - self.reg*qi)
        self.uid2i, self.iid2j = uid2i, iid2j
        return self

    def predict_user(self, user_id: str) -> pd.Series:
        ui = self.uid2i[user_id]
        scores = self.mu + self.bu[ui] + self.bi + self.P[ui] @ self.Q.T
        scores = np.clip(scores, 0.0, 5.0)
        return pd.Series(scores, index=self.items)


# 하이브리드 (User + Item + MF)

def hybrid_predict(df_user: pd.DataFrame,
                   df_item: pd.DataFrame,
                   mf_model: MF,
                   w_user: float = 0.33,
                   w_item: float = 0.33,
                   w_mf:   float = 0.34) -> pd.DataFrame:
    assert abs(w_user + w_item + w_mf - 1.0) < 1e-6
    users = df_user.index
    items = df_user.columns
    # MF 예측
    mf_scores = pd.DataFrame(index=users, columns=items, dtype=float)
    for u in users:
        mf_scores.loc[u] = mf_model.predict_user(u)
    # min-max 정규화 후 가중합
    def norm(df):
        lo = df.min().min(); hi = df.max().max()
        return (df - lo) / (hi - lo + 1e-8) if hi > lo else df*0
    U = norm(df_user); I = norm(df_item); M = norm(mf_scores)
    return w_user*U + w_item*I + w_mf*M

# 제약 필터

def _parse_allergy(val) -> List[str]:
    if isinstance(val, list):
        return [str(x) for x in val]
    if isinstance(val, str) and val.strip() and val.strip() != "[]":
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, (list, tuple)):
                return [str(x) for x in parsed]
        except Exception:
            pass
    return []


def combos_with_restriction_mask(combo_df: pd.DataFrame, restricted_ingredients: List[str]) -> pd.Series:
    if not restricted_ingredients:
        return pd.Series(False, index=combo_df.index)
    cols = [c for c in restricted_ingredients if c in combo_df.columns]
    if not cols:
        return pd.Series(False, index=combo_df.index)
    return (combo_df[cols].sum(axis=1) > 0)


# 추천 함수 (Soy-only + Allergy, no swap)

def recommend_for_user(user_id: str,
                       hybrid_scores: pd.DataFrame,
                       candidate_pool_n: int = 100,
                       final_k: int = 5,
                       diet_override: Optional[bool] = None,
                       diet_rank_mode: str = "weighted",  # "weighted" | "lexi"
                       w_score: float = 0.7,
                       w_cal: float = 0.3) -> Dict:
    assert user_id in hybrid_scores.index, f"Unknown user_id: {user_id}"

    # 사용자 제약
    is_veg = bool(df_users.loc[user_id].get("vegetarian", False)) if "vegetarian" in df_users.columns else False
    allergy_ids = _parse_allergy(df_users.loc[user_id].get("allergy", [])) if "allergy" in df_users.columns else []

    # 미평가 후보 (점수 내림차순)
    user_row = hybrid_scores.loc[user_id]
    unrated_mask = df_rating.loc[user_id].isna()
    candidates_by_score = user_row[unrated_mask].sort_values(ascending=False)

    # 알레르기 제외
    restr = list(allergy_ids)
    # 채식주의자면 비-소이 미트는 전부 제외
    if is_veg and SOY_ID is not None:
        restr.extend([m for m in MEAT_IDS if m != SOY_ID])
    restr_mask = combos_with_restriction_mask(df_combo, restr)
    candidates = candidates_by_score[~restr_mask.loc[candidates_by_score.index]]

    # 채식주의자면 '콩고기 단독 미트 조합'으로 후보 제한
    if is_veg and SOY_ONLY_IDS:
        candidates = candidates.loc[candidates.index.intersection(SOY_ONLY_IDS)]

    # 후보 상위 N
    pool_ids = candidates.index[:candidate_pool_n].tolist()
    pool_df = pd.DataFrame({
        "score": hybrid_scores.loc[user_id, pool_ids].values,
        "calories": combo_calories.loc[pool_ids].values
    }, index=pool_ids)

    # Diet 플래그
    if diet_override is not None:
        diet_flag = bool(diet_override)
    elif "diet" in df_users.columns:
        diet_flag = bool(df_users.loc[user_id].get("diet", False))
    else:
        diet_flag = False

    # 최종 정렬/선정
    if diet_flag:
        if diet_rank_mode == "weighted":
            eps = 1e-8
            s = pool_df["score"]; c = pool_df["calories"]
            z_s = (s - s.mean()) / (s.std(ddof=0) + eps)
            z_c_inv = - (c - c.mean()) / (c.std(ddof=0) + eps)
            utility = w_score * z_s + w_cal * z_c_inv
            final_k_df = pool_df.assign(utility=utility).sort_values("utility", ascending=False).head(final_k)
        elif diet_rank_mode == "lexi":
            final_k_df = pool_df.sort_values(["score", "calories"], ascending=[False, True]).head(final_k)
        else:
            raise ValueError("diet_rank_mode must be 'weighted' or 'lexi'")
    else:
        final_k_df = pool_df.sort_values("score", ascending=False).head(final_k)

    return {
        "user_id": user_id,
        "diet": diet_flag,
        "vegetarian": is_veg,
        "allergy": allergy_ids,
        "final_recommendations": final_k_df
    }

# Pretty print helpers — ingredients 포함 출력

def ingredient_ids_for(combo_id: str) -> List[str]:
    row = df_combo.loc[combo_id]
    return [c for c in df_combo.columns if int(row[c]) == 1]

def format_combo_line(combo_id: str, rec_row: pd.Series, diet_flag: bool) -> str:
    ings = ", ".join(s.lower() for s in ingredient_ids_for(combo_id))
    base = f"{combo_id} | score={rec_row['score']:.4f} | calories={rec_row['calories']:.1f}"
    if diet_flag and 'utility' in rec_row.index:
        base += f" | utility={rec_row['utility']:.6f}"
    return base + f" | ingredients: {ings}"

# 실행 (AUTO-RUN)
print("Building User-Based predictions...")
df_pred_user = predict_user_based(df_rating, k=30)

print("Building Item neighbors...")
item_neighbors = build_item_neighbors(df_rating, k=20)

print("Building Item-Based predictions...")
df_pred_item = predict_item_based(df_rating, item_neighbors)

print("Training MF model (biased MF)...")
mf = MF(factors=40, epochs=15, lr=0.01, reg=0.02).fit(df_rating)

print("Combining (hybrid: user+item+mf = 0.33/0.33/0.34)...")
df_pred_hybrid = hybrid_predict(df_pred_user, df_pred_item, mf, w_user=0.33, w_item=0.33, w_mf=0.34)

# 데모: 앞 5명
sample_users = df_rating.index[:5].astype(str).tolist()
for uid in sample_users:
    dflag = bool(df_users.loc[uid].get("diet", False)) if (uid in df_users.index and "diet" in df_users.columns) else False
    res = recommend_for_user(uid, df_pred_hybrid,
                             candidate_pool_n=100, final_k=5,
                             diet_override=None)  # override 금지
    rec_df = res["final_recommendations"]

    print("\n============================")
    print(f"[{uid}] (diet={dflag}) → Final Recommendations: 5")
    for cid, row in rec_df.iterrows():
        print(" ", format_combo_line(str(cid), row, dflag))
