import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import skops.io as sio
import os

# Create output directories if not exist
os.makedirs("Results", exist_ok=True)
os.makedirs("Model", exist_ok=True)

# 1. Load and shuffle data
drug_df = pd.read_csv("Data/drug200.csv")
drug_df = drug_df.sample(frac=1).reset_index(drop=True)

# 2. Split data into features (X) and target (y)
X = drug_df.drop("Drug", axis=1).values
y = drug_df["Drug"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=125
)

# 3. Define preprocessing pipeline
cat_col = [1, 2, 3]  # categorical: Sex, BP, Cholesterol
num_col = [0, 4]     # numerical: Age, Na_to_K

transform = ColumnTransformer(
    transformers=[
        ("encoder", OrdinalEncoder(), cat_col),
        ("num_imputer", SimpleImputer(strategy="median"), num_col),
        ("num_scaler", StandardScaler(), num_col),
    ]
)

pipe = Pipeline(
    steps=[
        ("preprocessing", transform),
        ("model", RandomForestClassifier(n_estimators=100, random_state=125)),
    ]
)

# 4. Train the model
pipe.fit(X_train, y_train)

# 5. Evaluate model
predictions = pipe.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")

# 6. Save metrics to file
with open("Results/metrics.txt", "w") as f:
    f.write(f"Accuracy = {round(accuracy, 2)}\n")
    f.write(f"F1 Score = {round(f1, 2)}\n")

# 7. Create and save confusion matrix
cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
disp.plot()
plt.tight_layout()
plt.savefig("Results/model_results.png", dpi=120)
plt.close()

# 8. Save the trained model
sio.dump(pipe, "Model/drug_pipeline.skops")
