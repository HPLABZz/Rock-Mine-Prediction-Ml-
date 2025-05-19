import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

sonar_data = pd.read_csv("sonar data-set.csv", header=None)

X = sonar_data.drop(columns=60, axis=1)
y = sonar_data[60]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training and testing sonar data using K-Neighbors MODEL 
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

knn_train_pred = knn_model.predict(X_train_scaled)
knn_test_pred = knn_model.predict(X_test_scaled)

knn_train_acc = accuracy_score(y_train, knn_train_pred)
knn_test_acc = accuracy_score(y_test, knn_test_pred)
knn_conf_matrix = confusion_matrix(y_test, knn_test_pred)

joblib.dump(knn_model, "models/knn_model.pkl")

# Training and testing sonar data using RandomForest Model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=4, random_state=42)
rf_model.fit(X_train_scaled, y_train)

rf_train_pred = rf_model.predict(X_train_scaled)
rf_test_pred = rf_model.predict(X_test_scaled)

rf_train_acc = accuracy_score(y_train, rf_train_pred)
rf_test_acc = accuracy_score(y_test, rf_test_pred)
rf_conf_matrix = confusion_matrix(y_test, rf_test_pred)

joblib.dump(rf_model, "models/random_forest_model.pkl")

joblib.dump(knn_model, "models/knn_model.pkl")

input_data = (0.021, 0.0121, 0.0203, 0.1036, 0.1675, 0.0418, 0.0723, 0.0828, 0.0494, 0.0686,
              0.1125, 0.1741, 0.271, 0.3087, 0.3575, 0.4998, 0.6011, 0.647, 0.8067, 0.9008,
              0.8906, 0.9338, 1, 0.9102, 0.8496, 0.7867, 0.7688, 0.7718, 0.6268, 0.4301,
              0.2077, 0.1198, 0.166, 0.2618, 0.3862, 0.3958, 0.3248, 0.2302, 0.325, 0.4022,
              0.4344, 0.4008, 0.337, 0.2518, 0.2101, 0.1181, 0.115, 0.055, 0.0293, 0.0183,
              0.0104, 0.0117, 0.0101, 0.0061, 0.0031, 0.0099, 0.008, 0.0107, 0.0161, 0.0133)

def predict_result_KNN():
    if len(input_data) == 60:
        input_array = np.asarray(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        pred_knn = knn_model.predict(input_scaled)
        pred_rf = rf_model.predict(input_scaled)
        # KNN Prediction
        if pred_knn[0]=='R':return "K-Neighbors Prediction: The object is Rock"   
        else:       return "K-Neighbors Prediction: The object is Mine"    
    else:
        return "Invalid input: Expected 60 features only."

def predict_result_RF():
    if len(input_data) == 60:
        input_array = np.asarray(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        pred_knn = knn_model.predict(input_scaled)
        pred_rf = rf_model.predict(input_scaled)
        # Random Forest Prediction
        if pred_rf[0]=='R':return "Random Forest Prediction: The object is Mine"
        else:       return "RandomForest Prediction: The object is Mine"
    else:
        print("Invalid input: Expected 60 features only.") 

if __name__ == "__main__":
    result = predict_result_KNN()
    print(result)
    result = predict_result_RF()
    print(result)
