'''
FS_random_forest_251216.py
avoid dataframe
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # FOR CLASSIFICATION
# FOR CLASSIFICATION
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

#%% data preprocessing
path = '/home/zzha/PycharmProjects/RampMerging4_250208/data/features/df_fea_tar_new251121.csv'
df = pd.read_csv(path)
print(df.columns)
print(df.head())
# filter out 'unknown' type
df_filtered = df.loc[df['state']!='unknown']
# convert state string into '0' or '1'
# following_mode = 1, free_mode = 0
df_filtered2 = df_filtered.copy()
df_filtered2['state'] = df_filtered['state'].replace({'following_mode': 1, 'free_mode': 0})
df_filtered3 = df_filtered2.loc[:,["id", "dis_to_pv", "v_pv", "dis_leaderAV", "size", "state"]]
print(df_filtered3)
# use numpy as input
data = df_filtered3.to_numpy()

#%% model training
# 2. Separate features and target variable
X = data[:, 1:5].astype(float)
y = data[:, 5].astype(int) # should be int instead of object

# 3. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create and train the Random Forest Regressor model
model = RandomForestClassifier(n_estimators=800, random_state=36)
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

#%%
# 6. Evaluate model performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 7. Feature importance analysis
importances = model.feature_importances_
feature_names = ["dis_to_pv", "v_pv", "dis_leaderAV", "size"]
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(importance_df)

#%% save the model
# joblib.dump(model, 'Models/follower_state_prediction_model_250415.pkl')
joblib.dump(model, '/home/zzha/PycharmProjects/RampMerging4_250208/models/follower_state_prediction_model_251121_ndarray.pkl')

#%% load model
# loaded_model = joblib.load('Models/follower_state_prediction_model_250415.pkl')
# loaded_model = joblib.load('Models/follower_state_prediction_model_250501.pkl')
loaded_model = joblib.load('/home/zzha/PycharmProjects/RampMerging4_250208/models/follower_state_prediction_model_251121_ndarray.pkl')
ls_new_features = [126.610488, 27.531120, 543.562369, 7]
# df_new_data = pd.DataFrame([ls_new_features],
#                         columns=['dis_to_pv', 'v_pv', 'dis_leaderAV', 'size'])
X_new = np.array(ls_new_features, dtype=float).reshape(1, -1)
new_state = loaded_model.predict(X_new)
print(new_state)

#%% plot
# Import necessary libraries
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Convert predicted probabilities to binary class (0 or 1)
y_pred_class = (y_pred >= 0.5).astype(int)

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred_class)

# Plot the confusion matrix with a color map
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')  # You can also try 'Greens', 'Purples', 'Oranges', etc.

# Add title
plt.title("Confusion Matrix")

# Show the plot
plt.show()
