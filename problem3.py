# importing Necessary Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Creating the Dataset

Data = {
    'ram_gb': [4, 8, 4, 16, 8, 8, 4, 16, 8, 16,
               4, 8, 4, 16, 8, 8, 4, 16, 8, 16,
               4, 8, 4, 16, 8, 8, 4, 16, 8, 16],

    'storage_gb': [256, 512, 128, 512, 256, 512, 256, 1024, 256, 512,
                   128, 512, 256, 1024, 256, 512, 128, 512, 256, 1024,
                   256, 512, 128, 512, 256, 512, 256, 1024, 256, 512],

    'processor_ghz': [2.1, 2.8, 1.8, 3.2, 2.4, 3.0, 2.0, 3.5, 2.6, 3.0,
                      1.6, 2.8, 2.2, 3.4, 2.5, 2.9, 1.9, 3.1, 2.3, 3.6,
                      2.0, 2.7, 1.7, 3.3, 2.4, 3.0, 2.1, 3.5, 2.6, 3.2],

    'price_inr': [28000, 45000, 22000, 72000, 38000, 52000, 26000, 95000, 42000, 68000,
                  20000, 48000, 29000, 88000, 40000, 50000, 23000, 70000, 36000, 98000,
                  25000, 46000, 21000, 75000, 39000, 53000, 27000, 92000, 43000, 73000]
}


df = pd.DataFrame(Data)

# Preparing the Model

x =np.array(df[['ram_gb', 'storage_gb', 'processor_ghz']])
y = np.array(df['price_inr'])

# Splitting the Data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print ("Features (X):")
print("RAM (GB), Storage (GB), Processor Speed (GHz)")
print (x)
print (f"\nShape : {x.shape} -> 3 features, 30 samples")

# Creating and Training the Model

model = LinearRegression()
model.fit(x_train, y_train)

print (f"Coefficients: {model.coef_}")
print (f"Intercept: {model.intercept_.round(2)}")


print (f"Coefficient for ram_gb contribute: {model.coef_[0]:.2f} INR per GB")
print (f"Coefficient for storage_gb contribute: {model.coef_[1]:.2f} INR per GB")
print (f"Coefficient for processor_ghz contribute: {model.coef_[2]:.2f} INR per GHz")

if model.coef_[0] > model.coef_[1] and model.coef_[0] > model.coef_[2]:
    print("RAM affects price more than Storage and Processor Speed.")

elif model.coef_[1] > model.coef_[0] and model.coef_[1] > model.coef_[2]:
    print("Storage affects price more than RAM and Processor Speed.")

else:
    print("Processor Speed affects price more than RAM and Storage.")

# Evaluating the Model

y_pred = model.predict(x_test)

print(f"\nR² Score: {r2_score(y_test, y_pred):.2f}")

# Making Predictions for New Data

new_laptop = np.array([[16, 512, 3.2]])  # 1 row, 3 columns
predicted_price = model.predict(new_laptop)

# printing the prediction details

print("\n--------- New Laptop details-----------:")

print("RAM: 16GB")
print("Storage: 512GB")
print("Processor Speed: 3.2GHz")

print(f"\nPredicted Price for Laptop with 16GB RAM, 512GB Storage, 3.2GHz Processor: INR {predicted_price[0]:.2f}")

# visualizing the Results

fig, axis = plt.subplots(1, 3, figsize=(18, 5))
features = ['ram_gb', 'storage_gb', 'processor_ghz']
name = ['RAM (GB)', 'Storage (GB)', 'Processor Speed (GHz)']
colors = ['blue', 'orange', 'green']

for i, (feature, name, color) in enumerate(zip(features, name, colors)):
    axis[i].scatter(df[feature], df['price_inr'], color=color, alpha=0.6)
    axis[i].set_xlabel(name)
    axis[i].set_ylabel('Price (INR)')
    axis[i].set_title(f'Price vs {name}')
    axis[i].grid(True, alpha=0.3)

plt.suptitle('Laptop Price Prediction based on Features', fontsize=16)
plt.tight_layout()

plt.savefig('laptop_price_prediction.png', dpi=150, bbox_inches='tight')
plt.show()

print (f"\n Plot saved as 'laptop_price_prediction.png'")

#Bonus question : Meera found a laptop with 8GB RAM, 512GB storage, 2.8 GHz for 55,000 INR. Is it overpriced?

print ("\n--- Bonus Question details ---")

print("Meera's Laptop:")
print("RAM: 8GB")
print("Storage: 512GB")
print("Processor Speed: 2.8GHz")

# Predicting price for Meera's laptop

meera_laptop = np.array([[8, 512, 2.8]])
predicted_price = model.predict(meera_laptop)

print(f"\nPredicted Price for Meera's Laptop: INR {predicted_price[0]:.2f}")

# Comparing with actual price

actual_price = 55000

print(f"Actual Price of Meera's Laptop: INR {actual_price}")

if predicted_price[0] > actual_price:
    print(f"Meera's laptop is underpriced by INR {predicted_price[0] - actual_price:.2f}")
else:
    print(f"Meera's laptop is overpriced by INR {actual_price - predicted_price[0]:.2f}")











