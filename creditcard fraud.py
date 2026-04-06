# Import Librarires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier, plot_importance
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import randint, uniform

# Load Data
file_path = '/Users/aadarshkarki/Downloads/creditcard.csv'
df = pd.read_csv(file_path)

# Preprocessing
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df[['Amount']])
df['scaled_time'] = scaler.fit_transform(df[['Time']])
df.drop(['Time', 'Amount'], axis=1, inplace=True)
df.insert(0, 'scaled_time', df.pop('scaled_time'))
df.insert(1, 'scaled_amount', df.pop('scaled_amount'))

# Visual: Fraud distribution
plt.figure(figsize=(10, 6))
sns.histplot(df[df['Class'] == 1]['scaled_time'], bins=50, kde=True, color='steelblue', alpha=0.7)
plt.title('Distribution of Fraud Transactions by Time')
plt.xlabel('Scaled Time')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Visual: KMeans + PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df.drop('Class', axis=1))
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_pca)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['cluster'], palette='Set2', alpha=0.6)
plt.title('KMeans Clustering (PCA Reduced)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()

# Train-Test Split
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

# Metric Function
def get_metrics(y_true, y_pred, y_proba=None):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'ROC AUC': roc_auc_score(y_true, y_proba if y_proba is not None else y_pred)
    }

# Baseline XGBoost
xgb_baseline = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_baseline.fit(X_resampled, y_resampled)
y_pred_baseline = xgb_baseline.predict(X_test)
y_proba_baseline = xgb_baseline.predict_proba(X_test)[:, 1]
baseline_metrics = get_metrics(y_test, y_pred_baseline, y_proba_baseline)

# Tuned XGBoost
scale_pos_weight = len(y_resampled[y_resampled == 0]) / len(y_resampled[y_resampled == 1])
param_dist = {
    'n_estimators': randint(150, 300),
    'max_depth': randint(4, 10),
    'learning_rate': uniform(0.01, 0.2),
    'subsample': uniform(0.7, 0.3),
    'colsample_bytree': uniform(0.7, 0.3),
    'gamma': uniform(0, 3),
    'scale_pos_weight': [scale_pos_weight],
    'reg_lambda': uniform(0, 3),
    'reg_alpha': uniform(0, 3)
}

xgb_tuned = RandomizedSearchCV(XGBClassifier(eval_metric='logloss', random_state=42),
    param_distributions=param_dist,
    n_iter=50, scoring='f1', cv=3, verbose=1, n_jobs=-1, random_state=42)
xgb_tuned.fit(X_resampled, y_resampled)
best_params = xgb_tuned.best_params_
xgb_model = XGBClassifier(**best_params, eval_metric='logloss', random_state=42)
xgb_model.fit(X_resampled, y_resampled)
y_pred_xgb = xgb_model.predict(X_test)
y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

# Anomaly Models
iso_model = IsolationForest(contamination=0.0017, random_state=42)
y_pred_iso = np.where(iso_model.fit_predict(X_test) == -1, 1, 0)
lof_model = LocalOutlierFactor(n_neighbors=20, contamination=0.0017)
y_pred_lof = np.where(lof_model.fit_predict(X_test) == -1, 1, 0)

# Results Table
results = pd.DataFrame({
    'Isolation Forest': get_metrics(y_test, y_pred_iso),
    'Local Outlier Factor': get_metrics(y_test, y_pred_lof),
    'XGBoost (Baseline)': baseline_metrics,
    'XGBoost (Tuned)': get_metrics(y_test, y_pred_xgb, y_proba_xgb)
}).T
print("\n Final Model Performance Comparison:\n")
print(results.round(4).to_string())

# Bar Chart
color_palette = {
    'Isolation Forest': '#F8766D',
    'Local Outlier Factor': '#B79F00',
    'XGBoost (Baseline)': '#7CAE00',
    'XGBoost (Tuned)': '#00BFC4'
}

for metric in results.columns:
    results[metric].plot(kind='bar', color=[color_palette[m] for m in results.index], title=metric)
    plt.ylabel(metric)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=15)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

# ROC Curve
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_proba_xgb)
fpr_base, tpr_base, _ = roc_curve(y_test, y_proba_baseline)
fpr_iso, tpr_iso, _ = roc_curve(y_test, y_pred_iso)
fpr_lof, tpr_lof, _ = roc_curve(y_test, y_pred_lof)

plt.figure(figsize=(8, 6))
plt.plot(fpr_xgb, tpr_xgb, label='XGBoost (Tuned)', color='#00BFC4')
plt.plot(fpr_base, tpr_base, label='XGBoost (Baseline)', color='#7CAE00')
plt.plot(fpr_iso, tpr_iso, label='Isolation Forest', color='#F8766D')
plt.plot(fpr_lof, tpr_lof, label='LOF', color='#B79F00')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve Comparison')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Confusion Matrix
conf_stats = {}
for name, preds in {
    'XGBoost (Tuned)': y_pred_xgb,
    'XGBoost (Baseline)': y_pred_baseline,
    'Isolation Forest': y_pred_iso,
    'Local Outlier Factor': y_pred_lof
}.items():
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    conf_stats[name] = {'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp}
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Fraud', 'Fraud'])
    cmap = LinearSegmentedColormap.from_list("custom_blues", ["#ddeeff", "#08306B"])
    disp.plot(cmap=cmap, values_format='d')
    plt.title(f'Confusion Matrix - {name}')
    plt.tight_layout()
    plt.show()

# Stacked Confusion Matrix Chart
conf_df = pd.DataFrame(conf_stats).T
conf_df[['TN', 'FP', 'FN', 'TP']].plot(
    kind='bar', stacked=True, figsize=(10, 6),
    color=['#C6DBEF', '#6BAED6', '#FD8D3C', '#E6550D']
)
plt.title('Confusion Matrix Components by Model')
plt.ylabel('Count')
plt.xticks(rotation=15)
plt.legend(title='Metric')
plt.tight_layout()
plt.show()

# Feature Importance
plot_importance(xgb_model, max_num_features=10, color='#00BFC4')
plt.title('Top 10 Features - XGBoost (Tuned)')
plt.tight_layout()
plt.grid(True)
plt.show()

# Radar Chart
labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

metrics_values = [
    results.loc[name].tolist() + [results.loc[name].iloc[0]]
    for name in results.index
]
colors = ['#F8766D', '#B79F00', '#7CAE00', '#00BFC4']
labels_model = list(results.index)

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for i in range(len(metrics_values)):
    ax.plot(angles, metrics_values[i], linewidth=2, label=labels_model[i], color=colors[i])
    ax.fill(angles, metrics_values[i], alpha=0.25, color=colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_yticklabels([])
ax.set_title("Model Comparison - Radar Chart", size=15)
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
plt.tight_layout()
plt.show()