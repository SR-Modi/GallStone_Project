# Carola- Explainability (SHAP + permutation importance)


# %% 1. Load data and the final model from Person 3
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path

P = Path("processed")
X_train = pd.read_csv(P / "X_train.csv")
X_test = pd.read_csv(P / "X_test.csv")
y_train = pd.read_csv(P / "y_train.csv").squeeze()
y_test = pd.read_csv(P / "y_test.csv").squeeze()

# The saved model was made with an older scikit-learn, so Colab (newer) cannot open it.
# Then we rebuild the SAME Random Forest, with the SAME settings as in
# DataProcessing_Complete.ipynb (cells 51-55).
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
try:
    model = joblib.load(P / "final_model.joblib")
    print("Loaded saved model")
except Exception as e:
    print("Could not load saved model -> retraining the same Random Forest.", type(e).__name__)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(RandomForestClassifier(n_estimators=300, random_state=42),
                        {"max_depth": [None, 4, 8], "min_samples_leaf": [1, 3, 5]},
                        cv=cv, scoring="roc_auc")
    grid.fit(X_train, y_train)
    print("Best settings:", grid.best_params_)
    model = grid.best_estimator_
    model.fit(X_train, y_train)
print(type(model))
print(X_train.shape, X_test.shape)
print(pd.read_csv(P / "ensemble_results.csv"))   # see which model won


from sklearn.metrics import roc_auc_score
print("Test AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
# Person 3's Random Forest Test AUC was 0.8506 -> it should be about 0.85.

# %% 2. SHAP values (global explanation)
explainer = shap.TreeExplainer(model)      # for RandomForest / XGBoost
sv = explainer(X_test)

# Random Forest gives 2 classes -> keep only the gallstone class
if sv.values.ndim == 3:
    sv = sv[:, :, 1]

# %% 3. Global plots
shap.plots.bar(sv, max_display=15, show=False)
plt.tight_layout(); plt.savefig("shap_bar.png", dpi=200); plt.show()

shap.plots.beeswarm(sv, max_display=15, show=False)
plt.tight_layout(); plt.savefig("shap_beeswarm.png", dpi=200); plt.show()

# %% 4. Dependence plots for the top 4 features
mean_abs = np.abs(sv.values).mean(axis=0)
top = X_test.columns[np.argsort(mean_abs)[::-1][:4]]
print("Top features:", list(top))
for f in top:
    shap.plots.scatter(sv[:, f], color=sv, show=False)
    plt.tight_layout(); plt.savefig(f"shap_dep_{f}.png", dpi=200); plt.show()

# %% 5. Local explanations (individual patients)
proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
yt = np.asarray(y_test)
cases = {
    "true_positive": np.where((pred == 1) & (yt == 1))[0][0],
    "true_negative": np.where((pred == 0) & (yt == 0))[0][0],
}
wrong = np.where(pred != yt)[0]
if len(wrong):
    cases["misclassified"] = wrong[0]
for name, i in cases.items():
    shap.plots.waterfall(sv[i], max_display=10, show=False)
    plt.title(f"{name} (patient index {i}, P(gallstone)={proba[i]:.2f})")
    plt.tight_layout(); plt.savefig(f"shap_local_{name}.png", dpi=200); plt.show()

# %% 6. Permutation importance
from sklearn.inspection import permutation_importance
pi = permutation_importance(model, X_test, y_test, scoring="roc_auc",
                            n_repeats=30, random_state=42)
pi_df = pd.DataFrame({"feature": X_test.columns,
                      "perm_mean": pi.importances_mean,
                      "perm_std": pi.importances_std}
                     ).sort_values("perm_mean", ascending=False)
print(pi_df.head(15))
top15 = pi_df.head(15).iloc[::-1]
plt.barh(top15.feature, top15.perm_mean, xerr=top15.perm_std)
plt.xlabel("Drop in ROC-AUC"); plt.tight_layout()
plt.savefig("perm_importance.png", dpi=200); plt.show()

# %% 7. Compare with Logistic Regression coefficients
from sklearn.linear_model import LogisticRegression
Xs_train = pd.read_csv(P / "X_train_scaled.csv")
logit = LogisticRegression(max_iter=5000).fit(Xs_train, y_train)
coef = pd.Series(logit.coef_[0], index=Xs_train.columns)

# NOTE: use Person 2's tuned settings (e.g. C, penalty) if you can.

# %% 8. One comparison table: SHAP vs permutation vs logistic
comp = pd.DataFrame({
    "shap_mean_abs": pd.Series(mean_abs, index=X_test.columns),
    "perm_importance": pd.Series(pi.importances_mean, index=X_test.columns),
    "logit_abs_coef": coef.abs(),
})
ranks = comp.rank(ascending=False)
ranks["avg_rank"] = ranks.mean(axis=1)
ranks = ranks.sort_values("avg_rank")
print(ranks.head(15))
ranks.to_csv("importance_comparison.csv")

# Do the methods agree? (Spearman rank correlation)
print(comp.corr(method="spearman"))

# %% 9. Stability check (small dataset -> are the rankings stable?)
# Re-compute SHAP on bootstrap resamples of the test set.
rng = np.random.default_rng(0)
tops = []
for _ in range(30):
    idx = rng.choice(len(X_test), len(X_test), replace=True)
    s = explainer(X_test.iloc[idx])
    v = s.values[:, :, 1] if s.values.ndim == 3 else s.values
    tops.append(pd.Series(np.abs(v).mean(0), index=X_test.columns).rank(ascending=False))
stab = pd.concat(tops, axis=1)
print(pd.DataFrame({"mean_rank": stab.mean(1), "std_rank": stab.std(1)})
      .sort_values("mean_rank").head(10))