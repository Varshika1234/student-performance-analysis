import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# =====================================================================
# 1. GENERATE SAMPLE DATASET
# =====================================================================
print("Generating dataset...")
np.random.seed(42)
num_students = 200

study_hours = np.random.uniform(1.0, 10.0, num_students)
attendance = np.random.uniform(60.0, 100.0, num_students)
# Create a final grade formula with some added random noise
final_grade = (study_hours * 4) + (attendance * 0.5) + np.random.normal(0, 3, num_students)
final_grade = np.clip(final_grade, 0, 100) # Keep grades within 0-100 range

df = pd.DataFrame({
    'Student_ID': range(1, num_students + 1),
    'Study_Hours': np.round(study_hours, 1),
    'Attendance_Percentage': np.round(attendance, 1),
    'Final_Grade': np.round(final_grade, 1)
})

# Save to CSV so you have the dataset file
csv_filename = "student_data.csv"
df.to_csv(csv_filename, index=False)
print(f"Dataset saved successfully as '{csv_filename}'")

# =====================================================================
# 2. EXPLORATORY DATA ANALYSIS & VISUALIZATION
# =====================================================================
print("\nAnalyzing data and generating insights...")
print(df.describe())

# Generate a correlation heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(df[['Study_Hours', 'Attendance_Percentage', 'Final_Grade']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Student Features')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

# Generate a scatter plot
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='Study_Hours', y='Final_Grade', hue='Attendance_Percentage', palette='viridis')
plt.title('Study Hours vs Final Grade')
plt.xlabel('Hours Studied per Day')
plt.ylabel('Final Grade (0-100)')
plt.tight_layout()
plt.savefig('study_hours_vs_grade.png')
plt.close()
print("Graphs saved as 'correlation_heatmap.png' and 'study_hours_vs_grade.png'")

# =====================================================================
# 3. MACHINE LEARNING MODEL (LINEAR REGRESSION)
# =====================================================================
print("\nTraining Machine Learning Model...")
X = df[['Study_Hours', 'Attendance_Percentage']]
y = df['Final_Grade']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Training Complete!")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared Score (Accuracy Variant): {r2:.2f}")

# Example prediction
sample_student = [[7.5, 85.0]] # 7.5 hours of study, 85% attendance
predicted_score = model.predict(sample_student)
print(f"\nPrediction Example:")
print(f"A student studying 7.5 hours with 85% attendance is predicted to score: {predicted_score[0]:.1f}")