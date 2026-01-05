'''
train RF-based prediction model
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import joblib

#%%
# 1. Data
# Feature data (platoon_type, distance, speed)
df_ft = pd.read_csv('/home/zzha/PycharmProjects/RampMerging3/data/df_combined_c_4f_241122.csv')
print(len(df_ft))
df_train_test = df_ft
# df_train_test = df_ft.iloc[4:8000,:]
df_train_test

#%%
# 2. Separate features and target variable
X = df_train_test[['platoon_type', 'dis_leader_pv', 'leader_v', 'leader_r_dis']]
y = df_train_test['tail_arr_duration']

# 3. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create and train the Random Forest Regressor model
model = RandomForestRegressor(n_estimators=800, random_state=36)
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

#%%
# 6. Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Mean Absolute Percentate Error: {mape:.4f}")
print(f"R-squared Score: {r2:.4f}")

#%% 7. Feature importance analysis
importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(importance_df)

#%% 8. Predict with new data example
ls_features = [7, 221.53, 21.76, 5.73]  # [3, 155.23, 24.74, 53.7524], [7, 127.04, 24.56, 5.98]
new_data = np.array([ls_features])  # Platoon type 1, distance 500 meters, speed 15 m/s
predicted_time = model.predict(new_data)
print(f"\nPredicted Arrival Time for new data: {predicted_time[0]:.2f} seconds")

#%% save the model
joblib.dump(model, '/home/zzha/PycharmProjects/RampMerging3/Models/m_arrival_prediction_model241122.pkl')

#%% load model
loaded_model = joblib.load('Models/m_arrival_prediction_model241122.pkl')
# new_data = np.array([[3, 155.23, 24.74, 53.7524]])  # Platoon type 1, distance 500 meters, speed 15 m/s
# ['platoon_type', 'dis_leader_pv', 'leader_v', 'leader_r_dis']
ls_new_features = [4, 83.05, 24.04, 184.59] # [3, 155.23, 24.74, 53.7524]
new_data = pd.DataFrame([ls_new_features],
                        columns=['platoon_type', 'dis_leader_pv', 'leader_v', 'leader_r_dis'])

predicted_time = loaded_model.predict(new_data)
print(f"\nPredicted Arrival Time for new data: {predicted_time[0]:.2f} seconds")

#%% Plot
import matplotlib.pyplot as plt
# plot two curves
plt.scatter(y_pred, y_test, s=2, marker='s', label='comparsion')
# title and label
plt.title('Comparison of predicted value and true value')
plt.xlabel('Predicted value (s)')
plt.ylabel('True value (s)')
# legend
plt.legend()
# grid
plt.grid(True)
# figure
plt.show()

