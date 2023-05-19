import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Step 2: Load the dataset of student academic performance metrics
data = pd.read_csv("student_performance.csv")

# Step 3: Clean the data if required

# Step 4: Evaluate the academic performance based on the given algorithm
a = data["1-1 semester percentage"]
b = data["1-2 semester percentage"]
c = data["2-1 semester percentage"]
d = data["2-2 semester percentage"]
e = data["3-1 semester percentage"]
f = data["3-2 semester percentage"]
g = data["Attendance percentage"]
h = data["extracurricular activities"]
i = data["Academic awards and achievements"]
j = data["Coding skills"]
k = np.array([a, b, c, d, e, f]).T

# Step 5: Calculate dropout
dropout = 1 if np.min(k) < 35 and g < 30 else 0

# Step 6: Calculate good performance
good_performance = 1 if np.all(k > 60) else 0

# Step 7: Calculate poor performance
poor_performance = 1 if np.max(k) < 40 else 0

# Step 8: Calculate support required
support_required = 1 if np.any((k >= 40) & (k < 60)) else 0

# Step 9: Calculate eligibility for placement
eligible_for_placement = 1 if (np.all(k > 65) and (j or i or h)) else 0

# Step 10: Print the results
print("Dropout:", dropout)
print("Good Performance:", good_performance)
print("Poor Performance:", poor_performance)
print("Support Required:", support_required)
print("Eligible for Placement:", eligible_for_placement)

# Step 11: Visualize the critical values as graphs across all students
plt.figure(figsize=(10, 6))
plt.plot(k)
plt.xlabel("Semester")
plt.ylabel("Percentage")
plt.title("Student Performance")
plt.legend(["1-1", "1-2", "2-1", "2-2", "3-1", "3-2"])
plt.show()

# Step 12: Prepare the data for the LSTM model
X = np.array([a, b, c, d, e, f]).T
y = np.array([dropout, good_performance, poor_performance, support_required])

# Step 13: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 14: Reshape the input data for LSTM modeling
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Step 15: Build and compile the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], 1)))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Step 16: Train the LSTM model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Step 17: Evaluate the LSTM model
_, train_accuracy = model.evaluate(X_train, y_train)
_, test_accuracy = model.evaluate(X_test, y_test)

print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)
