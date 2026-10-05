# Heart Disease Prediction

A classification project predicting whether a patient has heart disease from clinical measurements, comparing logistic regression, decision tree, random forest, SVM and KNN.

## Data

`heart.csv` contains 303 patients from the [UCI Cleveland heart disease study](https://archive.ics.uci.edu/dataset/45/heart+disease), in the version shared on Kaggle as the "Heart Attack Analysis & Prediction" dataset.

| Column | Description |
|---|---|
| age | Age in years |
| sex | 1 = male, 0 = female |
| cp | Chest pain type (4 categories) |
| trtbps | Resting blood pressure (mm Hg) |
| chol | Cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG result (3 categories) |
| thalachh | Maximum heart rate reached during exercise test |
| exng | Chest pain caused by exercise |
| oldpeak | ST depression during exercise compared to rest |
| slp | Slope of the ST segment at peak exercise (3 categories) |
| caa | Number of major blood vessels seen in the scan (0–3) |
| thall | Thallium stress test result (3 categories) |
| output | Target (see below) |

## Cleaning

- Removed 1 duplicated patient
- `caa = 4` (4 patients) and `thall = 0` (2 patients) aren't valid codes. They match the missing values reported in the original UCI data, so I dropped those 6 patients
- Checked the extreme values in cholesterol, blood pressure, max heart rate and oldpeak. They're unusual but medically possible, so I kept them
- **The target is flipped.** The dataset description says `output = 1` means more chance of a heart attack, but the `output = 0` group is older, has more blocked vessels, more exercise chest pain and a lower max heart rate. The 165/138 split also matches the healthy/sick counts in the original UCI data. I created `disease = 1 - output` so 1 means heart disease
- One-hot encoded `cp`, `restecg`, `slp` and `thall`, since they're categories, not amounts

296 patients remain after cleaning.

## Models

80/20 train/test split. Models were compared with 5-fold cross-validation on the training set; the test set was used once at the end. Logistic regression, SVM and KNN were scaled with `StandardScaler` inside a pipeline.

| Model | CV ROC AUC (default settings) |
|---|---|
| Logistic Regression | 0.915 |
| SVM | 0.911 |
| Random Forest | 0.897 |
| KNN | 0.883 |
| Decision Tree | 0.712 |

After tuning, logistic regression (0.920) and random forest (0.907) were tied, and a depth-limited decision tree reached 0.843.

**I chose logistic regression.** It scores the same as the random forest and its coefficients show how each feature affects the prediction.

## Results (test set, 60 patients)

| Metric | Score |
|---|---|
| ROC AUC | 0.919 |
| Accuracy | 87% |
| Recall (sick patients caught) | 79% (22 of 28) |
| False alarms | 2 of 32 healthy patients |

The test AUC matches cross-validation (0.92), so the model holds up on unseen patients. The tuned random forest scored slightly lower on the test set (0.896).

The strongest risk factors were the number of blocked vessels, a reversible defect on the thallium test, being male, exercise-induced chest pain and oldpeak. Chest pain types other than "asymptomatic", a normal thallium result and a higher max heart rate lowered the risk. Age had almost no effect once the other features were included.

## Limitations

- Small dataset (296 patients), so scores vary a lot between splits. The best regularisation setting for logistic regression also changed depending on the random split, while the model ranking stayed the same
- All patients come from one hospital in 1988, so the model may not generalise to other populations
- The category codes in this version differ from the original UCI codes and aren't documented, so labels like chest pain types were matched by their counts

## Next steps

- Lower the classification threshold to catch more sick patients
- Try gradient boosting (e.g. XGBoost)
- Test on the other UCI hospitals (Hungary, Switzerland, Long Beach)

## Running it

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Then open `model.ipynb` and run all cells.
