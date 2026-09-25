from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_curve, roc_auc_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from scipy.stats import spearmanr
from imblearn.over_sampling import SMOTE

ROOT = Path("analytics")
PLOTS = ROOT / "plots"
MODELS = ROOT / "models"
PLOTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
CSV = ROOT / "titanic.csv"

def load_once():
    if CSV.exists():
        return pd.read_csv(CSV)
    df = sns.load_dataset("titanic")
    df.to_csv(CSV, index=False)
    return df

def savefig(name):
    plt.tight_layout()
    plt.savefig(PLOTS / name, dpi=140)
    plt.close()

def main():
    df_raw = load_once()
    print("Shape:", df_raw.shape)
    print(df_raw.info())
    print(df_raw.describe(include="all").T)
    print("Missing %:")
    print((df_raw.isna().mean() * 100).loc[lambda s: s > 0])

    df = df_raw.copy()
    missing = (df.isna().mean() * 100)
    # Percentage-rule cleaning: <5% drop rows, 5-30% median/mode impute, very high missing drop.
    if missing["deck"] > 30:
        df = df.drop(columns=["deck"])
    if 5 <= missing["age"] <= 30:
        df["age"] = df["age"].fillna(df["age"].median())
    if 0 < missing["embarked"] < 5:
        df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
    if 0 < missing["embark_town"] < 5:
        df["embark_town"] = df["embark_town"].fillna(df["embark_town"].mode()[0])

    # Univariate
    for col in ["age", "fare"]:
        plt.figure(figsize=(7,4)); sns.histplot(df[col], kde=True); plt.title(f"{col.title()} distribution"); savefig(f"{col}_hist.png")
        plt.figure(figsize=(7,3)); sns.boxplot(x=df[col]); plt.title(f"{col.title()} boxplot"); savefig(f"{col}_boxplot.png")

    outlier_counts = {}
    for col in ["age", "fare"]:
        q1, q3 = df[col].quantile([.25,.75]); iqr = q3-q1
        outlier_counts[col] = int(((df[col] < q1-1.5*iqr) | (df[col] > q3+1.5*iqr)).sum())
    fare_mean, fare_median, fare_mode = df["fare"].mean(), df["fare"].median(), df["fare"].mode()[0]

    # Bivariate survival rates
    sex_rates = df.groupby("sex")["survived"].mean().sort_values(ascending=False)
    pclass_rates = df.groupby("pclass")["survived"].mean()
    sex_pclass_rates = df.groupby(["sex","pclass"])["survived"].mean()
    sex_rates.to_csv(ROOT/"survival_by_sex.csv")
    pclass_rates.to_csv(ROOT/"survival_by_pclass.csv")
    sex_pclass_rates.to_csv(ROOT/"survival_by_sex_pclass.csv")

    # Exact six-column correlation
    corr_cols = ["survived","pclass","age","sibsp","parch","fare"]
    corr = df[corr_cols].corr()
    plt.figure(figsize=(7,5)); sns.heatmap(corr, annot=True, fmt=".2f"); plt.title("Six-column correlation matrix"); savefig("correlation_heatmap.png")
    pairs=[]
    for i in range(len(corr_cols)):
        for j in range(i+1,len(corr_cols)):
            pairs.append((abs(corr.iloc[i,j]), corr.iloc[i,j], corr_cols[i], corr_cols[j]))
    strongest = sorted(pairs, reverse=True)[:2]

    # Four multivariate charts
    plt.figure(figsize=(7,4)); sns.barplot(data=df, x="pclass", y="survived", hue="sex"); plt.title("Survival rate by class and sex"); savefig("multivariate_1.png")
    plt.figure(figsize=(7,4)); sns.boxplot(data=df, x="pclass", y="fare", hue="survived"); plt.title("Fare by class and survival"); savefig("multivariate_2.png")
    plt.figure(figsize=(7,4)); sns.scatterplot(data=df, x="age", y="fare", hue="survived", style="sex", alpha=.7); plt.title("Age vs fare by survival and sex"); savefig("multivariate_3.png")
    plt.figure(figsize=(7,4)); sns.barplot(data=df, x="embarked", y="survived", hue="pclass"); plt.title("Survival by embarkation and class"); savefig("multivariate_4.png")

    # EDA standardization sanity check
    scaler = StandardScaler()
    for col in ["age","fare"]:
        z = scaler.fit_transform(df[[col]]).ravel()
        plt.figure(figsize=(7,4)); sns.histplot(z, kde=True); plt.title(f"{col.title()} after standardization (EDA only)"); savefig(f"{col}_standardized.png")

    # Classification
    target = "survived"
    features = ["pclass","sex","age","sibsp","parch","fare","embarked"]
    X, y = df[features], df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=42, stratify=y
    )
    numeric = ["pclass","age","sibsp","parch","fare"]
    categorical = ["sex","embarked"]
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical)
    ])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=42)
    }
    metrics=[]; roc_data={}
    for name, estimator in models.items():
        pipe=Pipeline([("preprocess",pre),("model",estimator)])
        pipe.fit(X_train,y_train)
        pred=pipe.predict(X_test); prob=pipe.predict_proba(X_test)[:,1]
        auc=roc_auc_score(y_test,prob)
        metrics.append({"model":name,"accuracy":accuracy_score(y_test,pred),
                        "precision":precision_score(y_test,pred,zero_division=0),
                        "recall":recall_score(y_test,pred,zero_division=0),
                        "f1":f1_score(y_test,pred,zero_division=0),"auc":auc})
        roc_data[name]=(y_test,prob)
        cm=confusion_matrix(y_test,pred)
        plt.figure(figsize=(4,3)); sns.heatmap(cm,annot=True,fmt="d"); plt.title(f"{name} confusion matrix"); plt.xlabel("Predicted"); plt.ylabel("Actual"); savefig(name.lower().replace(" ","_")+"_cm.png")
        if name=="Decision Tree":
            fitted_pre=pipe.named_steps["preprocess"]
            names=fitted_pre.get_feature_names_out()
            plt.figure(figsize=(18,9)); plot_tree(pipe.named_steps["model"],feature_names=names,class_names=["0","1"],filled=False,max_depth=4); plt.title("Decision Tree"); savefig("decision_tree.png")
    plt.figure(figsize=(7,5))
    for name,(yt,pr) in roc_data.items():
        fpr,tpr,_=roc_curve(yt,pr); plt.plot(fpr,tpr,label=f"{name} AUC={roc_auc_score(yt,pr):.3f}")
    plt.plot([0,1],[0,1],"--"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curves"); plt.legend(); savefig("roc_curves.png")

    # Imbalance comparison using logistic regression and train-only preprocessing
    base_pre = pre.fit(X_train, y_train)
    Xtr = base_pre.transform(X_train); Xte = base_pre.transform(X_test)
    variants = {}
    for label, clf, train_X, train_y in [
        ("baseline", LogisticRegression(max_iter=2000, random_state=42), Xtr, y_train),
        ("class_weight_balanced", LogisticRegression(max_iter=2000,class_weight="balanced",random_state=42), Xtr, y_train)
    ]:
        clf.fit(train_X,train_y); p=clf.predict(Xte)
        variants[label]=[precision_score(y_test,p),recall_score(y_test,p),f1_score(y_test,p)]
    sm=SMOTE(random_state=42)
    Xsm,ysm=sm.fit_resample(Xtr,y_train)
    clf=LogisticRegression(max_iter=2000,random_state=42).fit(Xsm,ysm)
    p=clf.predict(Xte)
    variants["SMOTE"]=[precision_score(y_test,p),recall_score(y_test,p),f1_score(y_test,p)]
    imbalance_df=pd.DataFrame(variants,index=["precision","recall","f1"]).T
    imbalance_df.to_csv(ROOT/"imbalance_comparison.csv")

    # RF GridSearch and OOB
    rf_pipe=Pipeline([("preprocess",pre),("model",RandomForestClassifier(oob_score=True,bootstrap=True,random_state=42,n_jobs=-1))])
    grid=GridSearchCV(rf_pipe, {"model__n_estimators":[150,250], "model__max_depth":[None,5,10], "model__max_features":["sqrt","log2"]},
                      cv=3, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train,y_train)
    best_rf=grid.best_estimator_
    oob=float(best_rf.named_steps["model"].oob_score_)

    # Save the complete preprocessing + model pipeline
    joblib.dump(best_rf, MODELS/"best_classifier_pipeline.joblib")

    # Reload the saved pipeline and demonstrate prediction on raw input
    reloaded_pipeline = joblib.load(MODELS/"best_classifier_pipeline.joblib")

    raw_sample = X_test.iloc[[0]].copy()
    reloaded_prediction = reloaded_pipeline.predict(raw_sample)

    print("Reloaded pipeline prediction on raw input:")
    print(reloaded_prediction)

    # Regression: predict fare from all other useful fields
    reg_features=["survived","pclass","sex","age","sibsp","parch","embarked"]
    Xr=df[reg_features]; yr=df["fare"]
    Xr_train,Xr_test,yr_train,yr_test=train_test_split(Xr,yr,test_size=.2,random_state=42)
    reg_pre=ColumnTransformer([
        ("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]),["survived","pclass","age","sibsp","parch"]),
        ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),["sex","embarked"])
    ])
    reg_pipe=Pipeline([("preprocess",reg_pre),("model",LinearRegression())])
    reg_pipe.fit(Xr_train,yr_train); rp=reg_pipe.predict(Xr_test)
    mae=mean_absolute_error(yr_test,rp); rmse=np.sqrt(mean_squared_error(yr_test,rp)); r2=r2_score(yr_test,rp)
    n=len(yr_test); p_features=reg_pipe.named_steps["model"].coef_.shape[0]
    adj=1-(1-r2)*(n-1)/(n-p_features-1)
    rho,pv=spearmanr(np.abs(yr_test.to_numpy()-rp),rp)
    hetero = "evidence of heteroscedasticity" if abs(rho)>=0.2 and pv<0.05 else "no strong evidence of heteroscedasticity"
    plt.figure(figsize=(7,4)); sns.scatterplot(x=rp,y=yr_test.to_numpy()-rp); plt.axhline(0,linestyle="--"); plt.xlabel("Predicted fare"); plt.ylabel("Residual"); plt.title("Regression residual plot"); savefig("residual_plot.png")

    clf_df=pd.DataFrame(metrics).sort_values("f1",ascending=False)
    clf_df.to_csv(ROOT/"classification_metrics.csv",index=False)
    reg_df=pd.DataFrame([{"model":"Multivariate Linear Regression","MAE":mae,"RMSE":rmse,"R2":r2,"Adjusted_R2":adj}])
    reg_df.to_csv(ROOT/"regression_metrics.csv",index=False)

    best_name=clf_df.iloc[0]["model"]
    report=f"""# Analytics Results

## EDA
- Raw shape: {df_raw.shape}
- Cleaned shape: {df.shape}
- Missing-value rule: columns below 5% missing are mode-imputed here because they are categorical; 5–30% numeric `age` is median-imputed; `deck` exceeds 30% and is dropped.
- IQR outliers — age: {outlier_counts['age']}; fare: {outlier_counts['fare']}.
- Fare mean={fare_mean:.2f}, median={fare_median:.2f}, mode={fare_mode:.2f}. The mean being above the median indicates right skew; the mode is lower than both, consistent with concentration at lower fares and a long high-fare tail.
- Survival by sex: {sex_rates.to_dict()}
- Survival by pclass: {pclass_rates.to_dict()}
- Strongest absolute correlations: {strongest[0][2]}–{strongest[0][3]} ({strongest[0][1]:.3f}) and {strongest[1][2]}–{strongest[1][3]} ({strongest[1][1]:.3f}).

## Multivariate chart interpretations
1. Class and sex jointly show substantial survival-rate differences; the sex gap remains visible within passenger classes.
2. Fare distributions differ strongly by class, while survival separates the distributions further, showing socioeconomic and class structure in the sample.
3. Age and fare do not form a simple linear cloud; survival and sex reveal additional structure, supporting multivariate modeling.
4. Embarkation survival differences are partly entangled with passenger class, illustrating why single-variable comparisons can hide confounding.

## Classification
All models use the same stratified 80/20 split. Preprocessing is fitted only on training data through `ColumnTransformer`.
The highest F1 on this fixed split is **{best_name}**. Metrics are in `classification_metrics.csv`.

## Imbalance
The three-way comparison is in `imbalance_comparison.csv`. SMOTE is applied only after train/test splitting and only to the transformed training fold.

## Tuning
Best RF parameters: `{str(grid.best_params_)}`.
Best tuned RF OOB score: {oob:.4f}.

## Regression
MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}, Adjusted R²={adj:.4f}.
Residual assessment: {hetero}. This conclusion is based on the residual-vs-prediction pattern and the association between absolute residual magnitude and predictions.

## Model comparison
Classification metrics and regression metrics are intentionally kept as separate metric groups because they are not directly comparable.
"""
    (ROOT/"ANALYTICS_REPORT.md").write_text(report,encoding="utf-8")
    print(report)

if __name__=="__main__":
    main()
