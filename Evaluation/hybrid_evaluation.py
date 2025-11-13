import pandas as pd
import numpy as np
import sys
import os

# 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(ROOT_DIR)

# 모델 import
from recommendation_model.Sandwich_Recommendation_System import (
    df_pred_hybrid,
    df_rating as df_rating_model
)

print("Hybrid predictions imported:", df_pred_hybrid.shape)

# 실제 rating 재로드
DATA_DIR = os.path.join(ROOT_DIR, "data")
df_rating = pd.read_csv(os.path.join(DATA_DIR, "rating_dataset.csv"), index_col="user_id")
print("Rating data loaded:", df_rating.shape)

# 공통 평가 항목(실제 300개 조합만)
COMMON_COLS = df_rating.columns.intersection(df_pred_hybrid.columns)
true_df = df_rating[COMMON_COLS]
pred_df = df_pred_hybrid[COMMON_COLS]

# RMSE / MAE
def calc_rmse_mae(true_df, pred_df):
    mask = true_df.notna()
    diff = true_df[mask] - pred_df[mask]
    rmse = np.sqrt((diff**2).mean().mean())
    mae = diff.abs().mean().mean()
    return float(rmse), float(mae)

# Precision@K / Recall@K
def precision_recall_at_k(true_df, pred_df, k=5, threshold=4.0):
    precisions, recalls = [], []
    for user in true_df.index:
        t = true_df.loc[user]
        p = pred_df.loc[user]
        good_items = t[t >= threshold].dropna().index
        topk = p.sort_values(ascending=False).index[:k]
        hit = len(set(topk) & set(good_items))
        precisions.append(hit / k)
        recalls.append(hit / max(len(good_items), 1))
    return float(np.mean(precisions)), float(np.mean(recalls))

# NDCG@K
def dcg(scores):
    return np.sum([(2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(scores)])

def ndcg_at_k(true_df, pred_df, k=5):
    ndcgs = []
    for user in true_df.index:
        t = true_df.loc[user]
        p = pred_df.loc[user]

        topk = p.sort_values(ascending=False).index[:k]
        rel_pred = t[topk].fillna(0).values

        ideal = t.sort_values(ascending=False).index[:k]
        rel_ideal = t[ideal].fillna(0).values

        ideal_dcg = dcg(rel_ideal)
        ndcg = dcg(rel_pred) / ideal_dcg if ideal_dcg > 0 else 0
        ndcgs.append(ndcg)
    return float(np.mean(ndcgs))

# 평가 실행
print("\n========== HYBRID MODEL EVALUATION ==========\n")

rmse, mae = calc_rmse_mae(true_df, pred_df)
precision5, recall5 = precision_recall_at_k(true_df, pred_df, k=5)
ndcg5 = ndcg_at_k(true_df, pred_df, k=5)

print(f"RMSE         : {rmse:.4f}")
print(f"MAE          : {mae:.4f}")
print(f"Precision@5  : {precision5:.4f}")
print(f"Recall@5     : {recall5:.4f}")
print(f"NDCG@5       : {ndcg5:.4f}")

print("\nEvaluation Completed.")