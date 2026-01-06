import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

N_COMPONENTS = 50

# ---------------- LOAD ----------------
train = pd.read_csv("cnn_embeddings_train.csv")
# test = pd.read_csv("cnn_embeddings_test.csv")

X_train = train.drop(columns=["id"]).values
# X_test = test.drop(columns=["id"]).values

# ---------------- SCALE (TRAIN ONLY) ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# ---------------- PCA ----------------
pca = PCA(n_components=N_COMPONENTS, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
# X_test_pca = pca.transform(X_test_scaled)

print("Explained variance:", pca.explained_variance_ratio_.sum())

# ---------------- SAVE ----------------
train_pca = pd.DataFrame(
    X_train_pca,
    columns=[f"visual_pc_{i+1}" for i in range(N_COMPONENTS)]
)
train_pca["id"] = train["id"]

# test_pca = pd.DataFrame(
#     X_test_pca,
#     columns=[f"visual_pc_{i+1}" for i in range(N_COMPONENTS)]
# )
# test_pca["id"] = test["id"]

train_pca.to_csv("cnn_visual_features_train.csv", index=False)
# test_pca.to_csv("cnn_visual_features_test.csv", index=False)

joblib.dump(scaler, "cnn_scaler.pkl")
joblib.dump(pca, "cnn_pca.pkl")