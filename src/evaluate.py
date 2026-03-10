import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, fbeta_score

def calculate_confusion_matrix_metrics(y_true, y_probs, thresholds):
    """Farklı eşik değerleri için confusion matrix bileşenlerini hesaplar."""
    results = []
    for thresh in thresholds:
        y_pred_thresh = (y_probs >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thresh).ravel()
        results.append({
            "Threshold": round(thresh, 1),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp
        })
    return pd.DataFrame(results)

def calculate_net_profit(df, v_cost, c_cost, r_rate, negative_impact_rate=0.0):
    """Confusion matrix dataframe'i üzerinden net kâr senaryosunu hesaplar."""
    total_churners = df['FN'] + df['TP'] 
    baseline_loss = total_churners * v_cost 

    cost_tp = (df['TP'] * c_cost) + (df['TP'] * (1 - r_rate) * v_cost)
    cost_fp = (df['FP'] * c_cost) + (df['FP'] * negative_impact_rate * v_cost)
    cost_fn = (df['FN'] * v_cost)
    
    total_model_cost = cost_tp + cost_fp + cost_fn
    return baseline_loss - total_model_cost
