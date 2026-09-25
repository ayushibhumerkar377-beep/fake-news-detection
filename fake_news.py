import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)



# 1. CREATE OUTPUT FOLDER


os.makedirs("Outputs", exist_ok=True)



# 2. TEXT CLEANING


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = text.strip()

    return text


# 3. LOAD DATASET


print("\n[*] Loading datasets...")

fake_data = pd.read_csv("Fake.csv")
true_data = pd.read_csv("True.csv")

# Labels
fake_data["label"] = 1
true_data["label"] = 0

# Combine both datasets
df = pd.concat(
    [fake_data, true_data],
    ignore_index=True
)

# Shuffle dataset
df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("[+] Dataset loaded successfully!")
print("[+] Total records:", len(df))



# 4. COMBINE TITLE + FULL TEXT


print("\n[*] Preparing text...")

df["title"] = df["title"].fillna("")
df["full_text"] = df["full_text"].fillna("")

df["text"] = (
    df["title"] + " " + df["full_text"]
)

df["cleaned_text"] = df["text"].apply(clean_text)



# 5. CLASS DISTRIBUTION


print("\n[*] Dataset distribution:")

print(
    df["label"].value_counts()
)

plt.figure(figsize=(6, 4))

sns.countplot(
    x=df["label"]
)

plt.title("Real vs Fake News Distribution")
plt.xlabel("Label (0 = Real, 1 = Fake)")
plt.ylabel("Number of Articles")

plt.tight_layout()

plt.savefig(
    "Outputs/class_distribution.png"
)

plt.show()



# 6. TRAIN TEST SPLIT


X_text = df["cleaned_text"]
y = df["label"]

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n[*] Train records:", len(X_train_text))
print("[*] Test records:", len(X_test_text))



# 7. TF-IDF


print("\n[*] Applying TF-IDF...")

vectorizer = TfidfVectorizer(
    max_features=2000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

print("[+] TF-IDF completed!")
print("[+] Number of features:", X_train.shape[1])



# 8. MACHINE LEARNING MODELS


models = {

    "KNN":
        KNeighborsClassifier(
            n_neighbors=5
        ),

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

    "Neural Network":
        MLPClassifier(
            hidden_layer_sizes=(50,),
            max_iter=200,
            random_state=42
        )
}



# 9. TRAIN AND EVALUATE


results = []
predictions = {}

for name, model in models.items():

    print("\n==============================")
    print("Training:", name)
    print("==============================")

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    predictions[name] = y_pred

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Real", "Fake"],
            zero_division=0
        )
    )



# 10. SAVE MODEL RESULTS


results_df = pd.DataFrame(results)

results_df.to_csv(
    "Outputs/model_comparison.csv",
    index=False
)

print("\n[+] Results saved!")



# 11. ACCURACY COMPARISON


plt.figure(figsize=(8, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="Accuracy"
)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Model")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "Outputs/accuracy_comparison.png"
)

plt.show()



# 12. ALL METRICS COMPARISON


metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

results_melted = results_df.melt(
    id_vars="Model",
    value_vars=metrics,
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=results_melted,
    x="Model",
    y="Score",
    hue="Metric"
)

plt.title("Model Performance Comparison")

plt.ylim(0, 1)

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "Outputs/all_metrics_comparison.png"
)

plt.show()



# 13. CONFUSION MATRICES


for name, y_pred in predictions.items():

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"]
    )

    plt.title(
        "Confusion Matrix - " + name
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    safe_name = name.replace(
        " ",
        "_"
    )

    plt.savefig(
        f"Outputs/confusion_matrix_{safe_name}.png"
    )

    plt.show()



# 14. RANDOM FOREST FEATURE IMPORTANCE


rf_model = models["Random Forest"]

feature_names = vectorizer.get_feature_names_out()

importance = rf_model.feature_importances_

feature_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

feature_df = feature_df.sort_values(
    "Importance",
    ascending=False
).head(20)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=feature_df,
    x="Importance",
    y="Feature"
)

plt.title(
    "Top 20 Important TF-IDF Features"
)

plt.tight_layout()

plt.savefig(
    "Outputs/feature_importance.png"
)

plt.show()



# 15. FINAL MESSAGE


print("\n===================================")
print(" FAKE NEWS DETECTION COMPLETED")
print("===================================")

print("\nGenerated files:")

print("1. class_distribution.png")
print("2. accuracy_comparison.png")
print("3. all_metrics_comparison.png")
print("4. confusion matrices")
print("5. feature_importance.png")
print("6. model_comparison.csv")

print("\nAll output files are saved inside:")
print("Outputs/")