# Analytics Results

## EDA
- Raw shape: (891, 15)
- Cleaned shape: (891, 14)
- Missing-value rule: columns below 5% missing are mode-imputed here because they are categorical; 5–30% numeric `age` is median-imputed; `deck` exceeds 30% and is dropped.
- IQR outliers — age: 66; fare: 116.
- Fare mean=32.20, median=14.45, mode=8.05. The mean being above the median indicates right skew; the mode is lower than both, consistent with concentration at lower fares and a long high-fare tail.
- Survival by sex: {'female': 0.7420382165605095, 'male': 0.18890814558058924}
- Survival by pclass: {1: 0.6296296296296297, 2: 0.47282608695652173, 3: 0.24236252545824846}
- Strongest absolute correlations: pclass–fare (-0.549) and sibsp–parch (0.415).

## Multivariate chart interpretations
1. Class and sex jointly show substantial survival-rate differences; the sex gap remains visible within passenger classes.
2. Fare distributions differ strongly by class, while survival separates the distributions further, showing socioeconomic and class structure in the sample.
3. Age and fare do not form a simple linear cloud; survival and sex reveal additional structure, supporting multivariate modeling.
4. Embarkation survival differences are partly entangled with passenger class, illustrating why single-variable comparisons can hide confounding.

## Classification
All models use the same stratified 80/20 split. Preprocessing is fitted only on training data through `ColumnTransformer`.
The highest F1 on this fixed split is **Random Forest**. Metrics are in `classification_metrics.csv`.

## Imbalance
The three-way comparison is in `imbalance_comparison.csv`. SMOTE is applied only after train/test splitting and only to the transformed training fold.

## Tuning
Best RF parameters: `{'model__max_depth': 5, 'model__max_features': 'sqrt', 'model__n_estimators': 150}`.
Best tuned RF OOB score: 0.8287.

## Regression
MAE=20.8977, RMSE=30.5328, R²=0.3975, Adjusted R²=0.3617.
Residual assessment: evidence of heteroscedasticity. This conclusion is based on the residual-vs-prediction pattern and the association between absolute residual magnitude and predictions.

## Model comparison
Classification metrics and regression metrics are intentionally kept as separate metric groups because they are not directly comparable.
