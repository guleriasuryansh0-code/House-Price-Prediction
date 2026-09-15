import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Creating the dataset
data = {
    "Area": [800, 1000, 1200, 1500, 1800, 2000, 2200, 2500],
    "Bedrooms": [2, 2, 3, 3, 4, 4, 4, 5],
    "Age": [15, 10, 8, 6, 5, 4, 3, 2],
    "Price": [42, 50, 62, 78, 95, 110, 125, 145]
}

df = pd.DataFrame(data)

print("House Price Dataset:")
print(df)


# Input features (X)
X = df[["Area", "Bedrooms", "Age"]]

# Target variable (y)
y = df["Price"]


# Splitting data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Creating Linear Regression model
model = LinearRegression()


# Training the model
model.fit(X_train, y_train)


# Predicting prices for test data
y_pred = model.predict(X_test)


# Calculating Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error:", mse)


# Predicting price of a new house
new_house = pd.DataFrame(
    [[1600, 3, 5]],
    columns=["Area", "Bedrooms", "Age"]
)

predicted_price = model.predict(new_house)

predicted_price = model.predict(new_house)
print("\nNew House Details:")
print("Area: 1600 sq.ft")
print("Bedrooms: 3")
print("Age: 5 years")

print("Predicted Price:", predicted_price[0], "Lakh")