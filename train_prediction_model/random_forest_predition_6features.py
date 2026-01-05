import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#%%
# 1. Data
# Feature data (platoon_type, distance, speed)
df_ft = pd.read_csv('data/df_combined_c_4f_102124.csv')
print(len(df_ft))

#%% handle with nan
df_ft = df_ft.dropna()
print(len(df_ft))
df_train_test = df_ft

#%%
# 2. Separate features and target variable
X = df_train_test[['platoon_type', 'dis_leader_pv', 'leader_v', 'leader_r_dis',
                   'tail_v', 'tail_r_dis']]
y = df_train_test['tail_arr_duration']

# 3. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create and train the Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

#%%
# 6. Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared Score: {r2:.2f}")

#%%
# 7. Feature importance analysis
importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(importance_df)

#%% 8. Predict with new data example
new_data = np.array([[3, 155.23, 24.74, 53.7524]])  # Platoon type 1, distance 500 meters, speed 15 m/s
predicted_time = model.predict(new_data)
print(f"\nPredicted Arrival Time for new data: {predicted_time[0]:.2f} seconds")

#%% save the model
import joblib
joblib.dump(model, 'm_arrival_prediction_model.pkl')


#%% load model
loaded_model = joblib.load('Models/m_arrival_prediction_model.pkl')
# new_data = np.array([[3, 155.23, 24.74, 53.7524]])  # Platoon type 1, distance 500 meters, speed 15 m/s
new_data = pd.DataFrame([[3, 155.23, 24.74, 53.7524]],
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

