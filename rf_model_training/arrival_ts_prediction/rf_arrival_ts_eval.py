import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import joblib


#%% function: data preprocess
def process_data(df):
    df = df.copy()
    # 1. drop nan
    df = df.dropna()
    # 2. modify 'AHH' to '3'
    df['platoon_type'] = df['platoon_type'].str.len()
    # 3. add target
    df['target'] = df['arrival_ts'] - df['prediction_ts']
    return df

#%% Feature data (platoon_type, distance, speed)
path = '/home/zzha/PycharmProjects/RampMerging_2026/data/features/df_rf_arrival_time_mixedHV_260830.csv'
df_ft = pd.read_csv(path)
print(df_ft.columns)
print(len(df_ft))

#%% Data preprocess
df_ft2 = process_data(df_ft)
print(df_ft2.columns)
print(len(df_ft2))

#%% Separate features and target variable
feature_cols = ['platoon_type', 'leader_to_pv_dis', 'leader_speed', 'leader_left_dis', 'm']

X = df_ft2[feature_cols].values
y = df_ft2['target'].values

#%% Load model
loaded_model = joblib.load('/home/zzha/PycharmProjects/RampMerging_2026/'
                           'rf_models/mr_arrival_prediction_model260319_ndarray.pkl')

#%% Make prediction
y_pred = loaded_model.predict(X)

#%% Evaluate model performance
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
rmse = mean_squared_error(y, y_pred, squared=False)
mape = mean_absolute_percentage_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Mean Squared Error: {mse:.4f}") # RMSE^2
print(f"Mean Absolute Error: {mae:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Mean Absolute Percentate Error: {mape:.4f}")
print(f"R-squared Score: {r2:.4f}")


#%% Plot
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Reproduce the original held-out evaluation figure from its paired results.
plot_data_path = Path('/home/zzha/PycharmProjects/RampMerging_2026/data/rf_arrival_time.csv')
plot_data = pd.read_csv(plot_data_path)
plot_pred = plot_data['y_pred'].to_numpy()
plot_true = plot_data['y_test'].to_numpy()
plot_rmse = mean_squared_error(plot_true, plot_pred) ** 0.5
plot_mae = mean_absolute_error(plot_true, plot_pred)
plot_r2 = r2_score(plot_true, plot_pred)

# Create figure and axis
fig, ax = plt.subplots(figsize=(5, 4), dpi=600)

textstr = (
    f"{'RMSE':<4}: {plot_rmse:.3f} s\n"
    f"{'MAE':<4}: {plot_mae:.3f} s\n"
    f"{'R²':<4}: {plot_r2:.3f}"
)

ax.text(
    0.95, 0.05,
    textstr,
    transform=ax.transAxes,
    fontsize=10,
    fontfamily='monospace',   # ✅ 核心
    horizontalalignment='right',
    verticalalignment='bottom',
    multialignment='left',
    # bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)

# Scatter plot: predicted vs true values
ax.scatter(
    plot_pred,
    plot_true,
    s=25,
    edgecolor='blue',
    facecolor='none',
    label='Prediction'
)

# Linear regression (first-order polynomial fit)
coefficients = np.polyfit(plot_pred, plot_true, 1)
fit_line = np.poly1d(coefficients)

# Determine plotting range automatically
min_val = min(plot_pred.min(), plot_true.min())
max_val = max(plot_pred.max(), plot_true.max())
upper = np.ceil(max_val / 10) * 10  # Round up to nearest 10 for better visualization

ax.set_xlim(0, 55)
ax.set_ylim(0, 55)

ax.set_xticks([0, 10, 20, 30, 40, 50, 55])
ax.set_yticks([0, 10, 20, 30, 40, 50, 55])

# Generate fitted line
x_fit = np.linspace(min_val, max_val, 600)
y_fit = fit_line(x_fit)

# Plot fitted line
ax.plot(x_fit, y_fit, 'r', label='Linear fit')

# Automatic margins for better visualization
ax.margins(0.05)

# Axis labels
ax.set_xlabel('Predicted value (s)')
ax.set_ylabel('True value (s)')

# Legend
ax.legend()

# Save the main figure as a 600 DPI LZW-compressed TIFF.
save_path = plot_data_path.parents[1] / "plot" / "plot_figures" / "rf_arrival_times2.tiff"
save_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(
    save_path, format="tiff", dpi=600, bbox_inches="tight",
    pil_kwargs={"compression": "tiff_lzw"},
)

# Display the figure
plt.show()

