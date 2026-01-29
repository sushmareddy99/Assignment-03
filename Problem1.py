# importing Necessary Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Creating the Dataset

Data = {
    "Ctr": [3.2, 5.8, 2.1, 7.4, 4.5, 6.2, 1.8, 5.1, 3.9, 8.5,
            4.2, 2.8, 6.8, 3.5, 7.1, 2.4, 5.5, 4.8, 6.5, 3.1,
            7.8, 2.6, 5.3, 4.1, 6.9, 3.7, 8.1, 2.2, 5.9, 4.6],

    'Total_views': [12000, 28000, 8500, 42000, 19000, 33000, 7000, 24000, 16000, 51000,
                    18000, 11000, 38000, 14500, 44000, 9500, 26000, 21000, 35000, 13000,
                    47000, 10000, 25000, 17500, 40000, 15000, 49000, 8000, 31000, 20000]
}


df = pd.DataFrame(Data)

# Preparing Data for Training

x = df[['Ctr']]
y = df['Total_views']

# Splitting into Training and Testing Sets

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Creating and Training the Model, Making Predictions

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# Making Prediction for new data

new_ctr = 8.0
predicted_views = model.predict([[new_ctr]])

print("\n----------Prediction for New Data:--------------")
print(f"Click-Through Rate (CTR): {new_ctr}%")
print(f"Predicted Total Views: {predicted_views[0]:.0f}")


# visualizing the Results

print(f" If Click-Through Rate is: {new_ctr}% then")
print(f"Expected Total Views : {predicted_views[0]:.0f}")

plt.figure(figsize=(10,6))
plt.scatter(df['Ctr'], df['Total_views'], color='blue', s=100, label='Actual Views')
plt.plot(df['Ctr'], model.predict(df[['Ctr']]), color='red', linewidth=2, label='Regression Line')
plt.scatter(new_ctr, predicted_views, color='green', s=200, marker='*', label='Predicted Views')

plt.xlabel('Click-Through Rate (%)', fontsize=12)   
plt.ylabel('Total Views', fontsize=12)

plt.title('Click-Through Rate vs Total Views Prediction', fontsize=14)

plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig('ctr_total_views_prediction.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nGraph saved as 'ctr_total_views_prediction.png'")

