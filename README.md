Models Included
Logistic Regression
Decision Tree
Random Forest
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)
Naive Bayes





----------------------

# Tech Stack
Python
Scikit-learn
NumPy
Pandas





# ---------------------------------------------------
# RESULTS DATAFRAME
# ---------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\n")
print("="*70)
print("FINAL MODEL COMPARISON")
print("="*70)

print(results_df)

# ---------------------------------------------------
# BEST MODEL
# ------------------------------------------------
-----------------------

anuzz--
