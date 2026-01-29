# importing Necessary Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Creating the Dataset

Data = {
    'distance_km': [2.5, 6.0, 1.2, 8.5, 3.8, 5.2, 1.8, 7.0, 4.5, 9.2,
                    2.0, 6.5, 3.2, 7.8, 4.0, 5.8, 1.5, 8.0, 3.5, 6.8,
                    2.2, 5.5, 4.2, 9.0, 2.8, 7.2, 3.0, 6.2, 4.8, 8.2],

    'prep_time_min': [10, 20, 8, 25, 12, 18, 7, 22, 15, 28,
                      9, 19, 11, 24, 14, 17, 6, 26, 13, 21,
                      10, 16, 14, 27, 11, 23, 12, 18, 15, 25],

    'delivery_time_min': [18, 38, 12, 52, 24, 34, 14, 45, 29, 58,
                          15, 40, 21, 50, 27, 35, 11, 54, 23, 43,
                          17, 32, 26, 56, 19, 47, 20, 37, 30, 53]
}

df = pd.DataFrame(Data)

# Preparing the Model

x = np.array(df[['distance_km', 'prep_time_min']])
y = np.array(df['delivery_time_min'])


# Splitting the Data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print ("Features (X):")
print("Distance (km) and Preparation Time (min)")
print (x)

print (f"\nShape : {x.shape} -> 2 features, 30 samples")

# Creating and Training the Model

model = LinearRegression()
model.fit(x_train, y_train)

print (f"Coefficients: {model.coef_}")
print (f"Intercept: {model.intercept_.round(2)}")

print (f"Coefficient for distance_km contribute: {model.coef_[0]:.2f} minutes per km")
print (f"Coefficient for prep_time_min contribute: {model.coef_[1]:.2f} minutes per min")

if model.coef_[0] > model.coef_[1]:
    print("Distance affects delivery time more than preparation time.")
else:
    print("Preparation time affects delivery time more than distance.")


# Evaluating the Model
y_pred = model.predict(x_test)

# Making Predictions for New Data

new_order = np.array([[7, 15]])  # 1 row, 2 columns
predicted_delivery_time = model.predict(new_order)

# Print the Prediction Result

print (f"------------- New Order Details -------------")
print (f"Distance (km): 7")
print (f"Preparation Time (min): 15")

print (f"Predicted Delivery Time for 7 km with 15 min prep time: {predicted_delivery_time[0]:.2f} minutes")

# visualizing the Results

fig, axis = plt.subplots(1, 2, figsize=(12, 5))
features = ['distance_km', 'prep_time_min']
names = ['Distance (km)', 'Preparation Time (min)']
colors = ['blue', 'orange']

for i, (feature, name, color) in enumerate(zip(features, names, colors)):
    axis[i].scatter(df[feature], df['delivery_time_min'], color=color, alpha=0.7)
    axis[i].set_xlabel(name)
    axis[i].set_ylabel('Delivery Time (min)')
    axis[i].set_title(f'Delivery Time vs {name}')
    axis[i].grid(True, alpha=0.3)

plt.suptitle('Delivery Time Prediction based on Distance and Preparation Time', fontsize=14, fontweight='bold')
plt.tight_layout()

plt.savefig('delivery_time_prediction.png', dpi=150, bbox_inches='tight')
plt.show()
print (f"\n Plot saved as 'delivery_time_prediction.png'")








