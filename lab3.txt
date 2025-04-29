#
#
# Your code to plot the anomalies for three different k values.
# Plot original series as well as residuals.
#
#
# Calculate the first (Q1) and third (Q3) quartiles of the residuals
Q1 = res.resid.quantile(0.25)
Q3 = res.resid.quantile(0.75)

# Calculate the IQR (Interquartile Range)
IQR = Q3 - Q1

# Define threshold multipliers (for 1.5, 2, and 3 IQR values)
threshold_1_5 = 1.5  # Common value for IQR anomaly detection
threshold_2 = 2  # 2 times IQR for stricter threshold
threshold_3 = 3  # 3 times IQR for an even stricter threshold

# Calculate the lower and upper bounds for anomalies based on IQR thresholds
lower_bound_1_5 = Q1 - threshold_1_5 * IQR
upper_bound_1_5 = Q3 + threshold_1_5 * IQR

lower_bound_2 = Q1 - threshold_2 * IQR
upper_bound_2 = Q3 + threshold_2 * IQR

lower_bound_3 = Q1 - threshold_3 * IQR
upper_bound_3 = Q3 + threshold_3 * IQR

# Detect anomalies by checking if residuals fall outside of the bounds
anomalies_1_5 = np.where((res.resid < lower_bound_1_5) | (res.resid > upper_bound_1_5), res.resid, np.nan)
anomalies_2 = np.where((res.resid < lower_bound_2) | (res.resid > upper_bound_2), res.resid, np.nan)
anomalies_3 = np.where((res.resid < lower_bound_3) | (res.resid > upper_bound_3), res.resid, np.nan)

# Create a figure and axis for the plot
fig, ax = plt.subplots(figsize=(16, 6))

# Plot original series (using some 'original_series' placeholder here)
ax.plot(df['timestamp'], df['daily_active_users'], label='Original Series', color='green', linestyle='-', linewidth=2)

# Plot residuals as grey dots
ax.plot(res.resid, marker='.', linestyle='none', label='Residuals', color='grey')

# Plot anomalies for 1.5*IQR threshold as orange circles
ax.plot(pd.Series(anomalies_1_5, res.resid.index), marker='o', linestyle='none', 
        label='1.5*IQR Anomalies', color='orange', fillstyle='none', markersize=8)

# Plot anomalies for 2*IQR threshold as red circles
ax.plot(pd.Series(anomalies_2, res.resid.index), marker='o', linestyle='none', 
        label='2*IQR Anomalies', color='red', fillstyle='none', markersize=10)

# Plot anomalies for 3*IQR threshold as purple circles
ax.plot(pd.Series(anomalies_3, res.resid.index), marker='o', linestyle='none', 
        label='3*IQR Anomalies', color='purple', fillstyle='none', markersize=12)

# Add labels and title
ax.set_title('Original Series and Anomalies Detected by IQR Method (1.5, 2, and 3 IQR thresholds)')
ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.legend()

# Add grid for better readability
ax.grid(True)

# Display the plot
plt.tight_layout()
plt.show()

