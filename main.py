import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from my_linear_regression import MyLinearRegression


def plot_data_and_fit(X, y, model):
    if X.shape[1] != 1:
        print('Plotting only works on single feature data.')
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Data', alpha=0.4)
    X_line = np.array([[X.min()], [X.max()]])
    y_line = model.predict(X_line)
    plt.plot(X_line, y_line, color='red', label='Fitted Line')
    plt.xlabel('Departure Time (in hours)')
    plt.ylabel('Time Doors Open (in hours)')
    plt.title('Linear Regression Line')
    plt.legend()
    plt.show()


def plot_cost_history(cost_history):
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, color='green', label='Cost (MSE)')
    plt.xlabel('Iteration')
    plt.ylabel('Cost (Mean Squared Error)')
    plt.title('Cost History During Training')
    plt.legend()
    plt.show()


def summarize_cost_history(cost_history, step=100):
    print("\nCost History Summary:")
    print(f"Initial Cost: {cost_history[0]:.6f}")
    for i in range(step, len(cost_history), step):
        print(f"Iteration {i}: Cost = {cost_history[i]:.6f}")
    print(f"Final Cost: {cost_history[-1]:.6f}")


# Create DataFrame from csv file, then create a deep copy for modification
metra_data_og = pd.read_csv('metra-door-times.csv')
metra_data = metra_data_og.copy(deep=True)

# Convert 'Date' to datetime
metra_data['Date'] = pd.to_datetime(metra_data['Date'])

# Convert 'Departure' and 'Doors Open' to datetime
metra_data['Departure'] = pd.to_datetime(metra_data['Departure'], format='%I:%M:%S %p').dt.time
metra_data['Doors Open'] = pd.to_datetime(metra_data['Doors Open'], format='%I:%M:%S %p').dt.time

# Convert 'Departure' and 'Doors Open' to floats
metra_data['Departure'] = (pd.to_datetime(metra_data['Departure'], format='%H:%M:%S').dt.hour +
                           pd.to_datetime(metra_data['Departure'], format='%H:%M:%S').dt.minute / 60)
metra_data['Doors Open'] = (pd.to_datetime(metra_data['Doors Open'], format='%H:%M:%S').dt.hour +
                            pd.to_datetime(metra_data['Doors Open'], format='%H:%M:%S').dt.minute / 60)

# Add a 'Day' column, then reorder the DataFrame
metra_data['Day'] = metra_data['Date'].dt.dayofweek  # or dt.day_name() for string names
reorder = ['Date', 'Day', 'Departure', 'Doors Open', 'Difference']
metra_data = metra_data[reorder]

print(f"Original DataFrame head(5):\n{metra_data_og.head(5)}")
print(f"\nUpdated DataFrame head(5):\n{metra_data.head(5)}")

# Visualization
# Departure vs Doors Open
plt.figure(figsize=(10, 6))
plt.scatter(metra_data['Departure'], metra_data['Doors Open'], alpha=0.4, color='magenta')
plt.title('Departure Time vs Doors Opening')
plt.xlabel('Departure Time (hours)')
plt.ylabel('Doors Open (hours)')

plt.show()

X = metra_data[['Departure']]  # Features
y = metra_data['Doors Open']  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MyLinearRegression(learning_rate=0.001, num_iters=1000)
model.fit(X_train.values, y_train.values.reshape(-1, 1))
predictions = model.predict(X_test.values)

sk_model = LinearRegression()
sk_model.fit(X_train.values, y_train.values.reshape(-1, 1))
sk_pred = sk_model.predict(X_test.values)

plot_data_and_fit(X.values, y.values.reshape(-1, 1), model)
plot_cost_history(model.cost_history)
summarize_cost_history(model.cost_history)

print("\nPredictions: (MyLinearRegression, Sklearn) | Target")
for i in range(len(y_test)):
    print(f"({predictions[i]}, {sk_pred[i]}) | {y_test.values[i]}")

mse = mean_squared_error(y_test.values, predictions.flatten())
print(f"\nMSE: {mse}")

my_mse = model.score(X_test.values, y_test.values.reshape(-1, 1))
print(f"My MSE: {my_mse * 2}")  # Sklearn uses 1/m to compute cost instead of 1/2m I used in my calculation

