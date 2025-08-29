import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

# Load CSV from data folder
csv_path = os.path.join(os.path.dirname(__file__), '/storage/emulated/0/pyfiles/MLD3/csv/Pokemon.csv')
df = pd.read_csv(csv_path)

# Select features and target
X = df[['Attack', 'Defense', 'Speed']]
y = df['Legendary']  # binary target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Create plots folder if not exists
plots_dir = os.path.join(os.path.dirname(__file__), '/storage/emulated/0/pyfiles/MLD3/plots')
os.makedirs(plots_dir, exist_ok=True)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"))
plt.close()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0,1],[0,1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig(os.path.join(plots_dir, "roc_curve.png"))
plt.close()

# Classification report
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Plots saved in plots/ folder!")