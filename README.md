# Human Activity Recognition from Smartphone Sensors

Classifies what a person is doing (walking, walking upstairs, walking downstairs, sitting, standing, laying) from 2.56-second windows of smartphone accelerometer and gyroscope data, using the [UCI Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) dataset. Three models are compared, the best is saved with joblib, and a one-screen Streamlit app lets you inspect individual predictions.

**Live app:** _link added after deployment_

| Model | Tuned setting | CV accuracy (train only) | **Test accuracy** |
|---|---|---|---|
| **Logistic regression (best)** | `C = 0.1` | 93.5% ± 4.0% | **95.0%** |
| Random forest | `n_estimators = 25` | 91.9% ± 3.9% | 91.7% |
| Decision tree | `max_depth = 6` | 86.6% ± 2.9% | 85.5% |

---

## Quick start

Requires **Python 3.12** (the version the pinned requirements were tested with).

```bash
git clone <this repo>
cd <repo folder>
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py               # opens http://localhost:8501
```

The trained model (`models/`) and the test set the app uses (`app_data/`) are included, so the app runs without downloading the raw dataset.

### Re-running the full pipeline

1. Download the dataset zip from the UCI link above and unzip it so this path exists: `data/UCI HAR Dataset/train/X_train.txt` (the download is a zip inside a zip, so unzip both).
2. Run each step from the project root:

```bash
python -m src.data_loader        # sanity check: shapes and subject split
python -m src.explore            # missing values + class balance
python -m src.train              # tune 3 models with CV, save best (~2 min)
python -m src.evaluate           # test-set metrics, confusion matrices, top features
python -m src.confidence         # choose the app's uncertainty threshold
python -m src.prepare_app_data   # export test set for the app
python -m tests.test_app_samples # test the app on 10 samples
```

## Project structure

```
app.py                      Streamlit app
src/data_loader.py          Loads train/test, fixes duplicate feature names, maps labels
src/explore.py              Missing-value and class-balance checks
src/train.py                GridSearchCV + GroupKFold tuning; saves all models + best
src/evaluate.py             Test metrics, confusion matrices, top confusions, top-10 features
src/confidence.py           Picks the "uncertain" threshold from out-of-fold predictions
src/prepare_app_data.py     Writes app_data/test_set.parquet
tests/test_app_samples.py   Drives the real app on 10 samples and checks what it shows
models/                     best_model.joblib (+ metadata) and the other two models
reports/                    All CSV outputs and confusion-matrix images
```

---

## Data checks and decisions

| Decision | Reason |
|---|---|
| **Kept the provided train/test split** (21 volunteers train, 9 test, no overlap) | The split is by person, so test accuracy measures performance on *new people*. Re-splitting randomly would put the same person in both sets and inflate scores (leakage). |
| **No imputation** | 0 missing values in 4.1M training and 1.7M test cells; all labels mapped. |
| **No class weighting or resampling** | Classes are reasonably balanced: 13.4%–19.1% each in train (largest/smallest = 1.43) and a similar mix in test (1.28). Accuracy is a fair headline metric; per-class precision/recall are reported anyway. |
| **Renamed 42 duplicated feature names** (e.g. `..._dup1`) | `features.txt` repeats some `bandsEnergy()` names; pandas needs unique column names. |
| **Cross-validation with `GroupKFold(5)` grouped by subject** | Mirrors the test setup: each validation fold contains only unseen people. Plain `KFold` would leak a person's data across folds and overestimate accuracy. |
| **One hyperparameter per model**: LR `C`, tree `max_depth`, forest `n_estimators` | Required by the assignment; each controls model complexity. |
| **`StandardScaler` only for logistic regression, inside a `Pipeline`** | LR is scale-sensitive; trees are not. Inside a Pipeline the scaler is re-fit on each training fold, so validation data never leaks into it. |
| **Best model chosen by CV accuracy, not test accuracy** | The test set is used once, at the end, only for reporting. |
| **`random_state=42`, pinned library versions** | Reproducibility; joblib models should be loaded with the scikit-learn version that saved them (recorded in `models/best_model_meta.json`). |

**Tuning results** (`reports/cv_tuning_curves.csv`): the decision tree shows classic overfitting: CV accuracy peaks at depth 6–8 (86.6%) and falls to 84.7% with unlimited depth. LR peaks at `C = 0.1`. The random forest's curve is flat (91.6–91.9% for 25–400 trees), well within the ±4% fold-to-fold variation, so the number of trees is not what limits it. The large fold-to-fold spread itself shows that **some people are much harder to classify than others**.

## Test-set evaluation

Per-activity precision / recall (precision: when the model says X, how often is it right; recall: of all true X, how many it finds):

| Activity | Logistic regression | Random forest | Decision tree |
|---|---|---|---|
| Walking | 0.94 / 0.99 | 0.86 / 0.96 | 0.86 / 0.93 |
| Walking upstairs | 0.95 / 0.95 | 0.88 / 0.85 | 0.87 / 0.73 |
| Walking downstairs | 0.99 / 0.94 | 0.95 / 0.86 | 0.75 / 0.82 |
| Sitting | 0.96 / **0.88** | 0.92 / 0.88 | 0.89 / **0.70** |
| Standing | **0.88** / 0.96 | 0.90 / 0.93 | 0.77 / 0.92 |
| Laying | 1.00 / 0.97 | 1.00 / 1.00 | 1.00 / 1.00 |

Confusion matrices: [logistic regression](reports/confusion_matrix_logistic_regression.png) · [random forest](reports/confusion_matrix_random_forest.png) · [decision tree](reports/confusion_matrix_decision_tree.png)

### Which activities get confused, and why

The most common mistakes (`reports/top_confusions.csv`) are the same for every model:

1. **Sitting → standing** is the largest error for every model: 11.8% of sitting samples for LR, 11.6% for the forest, 29.9% for the tree. Low sitting recall plus low standing precision is the same error seen from both sides.
2. **Walking upstairs ↔ walking ↔ walking downstairs**: 5–18% depending on model.
3. **Laying** is almost never confused (100% recall for both tree models).

**Hypothesis:** sitting and standing are both *still, upright* postures with the phone at the waist. With no movement to tell them apart, the only clue is a slight difference in hip tilt, which changes how gravity falls on the phone's axes. The data support this: the mean Y-axis gravity feature averages +0.12 when sitting vs −0.18 when standing (scaled units), a small gap compared with how differently people sit (upright, slouched, leaning back). The three walking activities share the same rhythmic gait pattern and differ mainly in subtle vertical-acceleration and axis-correlation patterns. Laying is easy because lying down rotates the phone about 90°, flipping the direction of gravity entirely.

## Top 10 features of the best model (logistic regression)

**Decoding the names:** `t` = time domain, `f` = frequency domain (after FFT: rhythm and regularity) · `Body` = the person's movement, `Gravity` = direction of "down" relative to the phone (tilt) · `Acc` = accelerometer, `Gyro` = gyroscope (rotation) · `Jerk` = how suddenly the signal changes · `X/Y/Z` = phone axes · `angle(A, gravityMean)` = angle between an axis and the average gravity direction.

Importance = mean absolute coefficient across the six classes. Inputs are standardized, so the coefficients are comparable across features.

| # | Feature | What it measures in plain English | What it separates |
|---|---|---|---|
| 1 | `tBodyGyroJerk-entropy()-X` | How irregular and unpredictable sudden twisting motions are | Moving (high) vs still (low) |
| 2 | `tGravityAcc-mean()-Y` | Average tilt of the phone along its Y axis | Sitting vs standing; laying |
| 3 | `tGravityAcc-max()-Y` | Same tilt, highest value in the window | Sitting vs standing; laying |
| 4 | `tGravityAcc-min()-Y` | Same tilt, lowest value in the window | Sitting vs standing; laying |
| 5 | `angle(Y,gravityMean)` | Angle between the phone's Y axis and "down" | Sitting (~0) vs standing/walking (~0.2) |
| 6 | `fBodyGyro-entropy()-X` | How irregular the rhythm of rotation is | Moving vs still |
| 7 | `tBodyAcc-correlation()-X,Y` | Whether movement along two axes happens together | Stairs (more negative) vs flat walking |
| 8 | `tGravityAcc-energy()-Y` | How strongly gravity falls on the Y axis | Laying vs everything else |
| 9 | `angle(X,gravityMean)` | Angle between the phone's X axis and "down" | Upright vs lying down |
| 10 | `tGravityAcc-max()-X` | Peak gravity along X (X points roughly down when upright) | Upright vs lying down |

In short, the model asks three questions: *Is the person moving?* (#1, #6), *Are they lying down?* (#8–#10), and *Sitting or standing?* (#2–#5, small tilt differences). Caveat: #2–#4 are nearly identical signals (gravity barely changes within 2.5 s). Logistic regression spreads weight across correlated features, so their individual ranks are not meaningful; the robust conclusion is "Y-axis tilt matters".

## The app

Pick a test sample (filter by true activity, show only mistakes or uncertain predictions, or press 🎲). The app shows the **predicted activity**, the **model's confidence** (probability of the predicted class), the **true label**, a ✅/❌ banner, and a bar chart of all six probabilities.

### What the app does when the model is uncertain

If confidence is **below 80%**, the app shows a ⚠️ warning with the top two candidate activities and tells the user to treat the result as "needs more data", not a final answer.

The threshold was chosen **from training data only**, using out-of-fold predictions (each sample predicted by a model that never saw that person) in `src/confidence.py`:

| Confidence | 0–50% | 50–60% | 60–70% | 70–80% | 80–90% | 90–100% |
|---|---|---|---|---|---|---|
| Accuracy | 52% | 55% | 59% | 69% | 82% | 97% |

The model's confidence roughly matches its actual accuracy, so it is a usable signal. Below 80%, predictions are right only about 61% of the time. Flagging them affects 8% of samples, catches 49% of all errors, and leaves the remaining predictions at 96% accuracy.

### Test on 10 samples (`tests/test_app_samples.py`)

The test drives the real app with Streamlit's `AppTest`, sets each sample, and checks that the displayed prediction, confidence, banners, and warning all agree with the model.

| Test row | True label | Predicted | Confidence | App shows |
|---|---|---|---|---|
| 1089 | Sitting | Standing | 100% | ❌ incorrect, **no warning** |
| 1122 | Laying | Standing | 91% | ❌ incorrect, **no warning** |
| 2232 | Standing | Sitting | 78% | ❌ incorrect + ⚠️ uncertain |
| 2524 | Walking downstairs | Walking upstairs | 52% | ❌ incorrect + ⚠️ uncertain |
| 803 | Sitting | Sitting | 60% | ✅ correct + ⚠️ uncertain |
| 53 | Sitting | Sitting | 90% | ✅ correct |
| 294 | Walking upstairs | Walking upstairs | 98% | ✅ correct |
| 722 | Walking downstairs | Walking downstairs | 100% | ✅ correct |
| 254 | Walking | Walking | 100% | ✅ correct |
| 2767 | Standing | Standing | 100% | ✅ correct |

**Takeaways:** the warning correctly flags uncertain errors (rows 2232, 2524), but it **cannot catch confident mistakes** (row 1089 was 100% sure and wrong). It also sometimes flags correct answers (row 803). Low confidence is a useful warning; high confidence is not a guarantee. A real product should combine predictions over several consecutive windows rather than trusting any single 2.56-second window.

## Limitations

1. **Narrow participant group.** 30 volunteers aged 19–48, described as healthy. Older adults, children, and people with mobility conditions (walker use, limp, Parkinson's) move differently, so accuracy will likely drop for exactly the groups activity monitoring often targets. Even within this group, CV accuracy varied by ±4% depending on which people were held out.
2. **One phone, one placement.** Every recording used a Samsung Galaxy S II worn on the waist. Seven of the top 10 features measure the phone's tilt relative to gravity, so a phone in a pocket, bag, or hand would change their meaning. The sitting/standing distinction, which already relies on a small hip-tilt difference, would probably fail first.
3. **Scripted lab data, not real life.** Activities followed a protocol in a controlled setting and were labeled from video, and each sample is a clean pre-cut window of a single activity. Real use includes transitions, mixed activities (walking while texting), and activities outside the six classes (running, cycling, driving), which this model would force into one of its six labels.

Also note: the 561 features were engineered by the dataset authors, so results depend on their preprocessing. Windows overlap by 50%, so neighbouring samples are not fully independent (this doesn't leak across the person-based split).

## Deployment

Platform limits were checked on 2026-11-14:

- **Hugging Face Spaces:** a Streamlit app must run as a Docker Space, and [Docker and Gradio Spaces now require a paid plan](https://huggingface.co/docs/hub/spaces-overview) (free accounts get static Spaces and limited ZeroGPU Gradio Spaces), so it was not used.
- **Streamlit Community Cloud (used):** free. Per the [docs](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app), apps get up to about 2.7 GB of memory and go to sleep after 12 hours without traffic (anyone can wake them). Only [Python versions still receiving security updates](https://docs.streamlit.io/deploy/streamlit-community-cloud/status) are offered, which is why the project uses Python 3.12 rather than 3.9. **Select Python 3.12 under "Advanced settings" when deploying**, because the version can't be changed afterwards. This app uses about 15 MB of data and a 60 KB model, far below the limits.

## Source

Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013). *A Public Domain Dataset for Human Activity Recognition Using Smartphones.* ESANN 2013. Dataset: UCI Machine Learning Repository, CC BY 4.0.
