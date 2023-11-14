import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import joblib
from sklearn.impute import SimpleImputer


file_path = 'DataProcessed.xlsx'
household_df = pd.read_excel(file_path,sheet_name='houseprice')
household_org = household_df.copy()

# Convert 'Household Price' to float and handle missing values
household_df['Household Price'] = household_df['Household Price'].replace('ND', 0).astype(float)
mean_household_price = household_df['Household Price'].mean()
household_df['Household Price'].fillna(mean_household_price, inplace=True)

# One-hot encode categorical variables
household_df = pd.get_dummies(household_df, columns=['Property Type', 'Negeri', 'Daerah'])

# Define the features (X) and the target variable (y)
X = household_df.drop(columns=['Household Price'])
y = household_df['Household Price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Regressor model
model = DecisionTreeRegressor(random_state=42)

# Fit the model to the training data
model.fit(X_train, y_train)

# Save the trained model using joblib
joblib.dump(model, 'decision_tree_model.pkl')

# Define a function to make predictions
def predict_household_price(new_data):
    # One-hot encode categorical variables in the new data
    new_data_df = pd.get_dummies(new_data, columns=['Property Type', 'Negeri', 'Daerah'])

    new_data = pd.DataFrame(0, columns=X.columns, index=[0])

    for col in new_data_df.columns:
      if col in new_data.columns:
        new_data.loc[0, col] = new_data_df.loc[0, col]

    # Make predictions using the trained model
    predicted_price = model.predict(new_data)
    return predicted_price[0]

# Streamlit UI
st.title("Household Price Prediction App")

unique_property_types = household_org['Property Type'].unique()
unique_negeri = household_org['Negeri'].unique()
unique_daerah = household_org['Daerah'].unique()

# Collect user input for prediction
year = st.selectbox("Year", ['2023','2024','2025','2026','2027','2028'])  # Adjust the year range as needed
property_type = st.selectbox("Property Type", unique_property_types)
negeri = st.selectbox("Negeri", unique_negeri)
daerah = st.selectbox("Daerah", unique_daerah)

# Create a DataFrame with user input
user_input = pd.DataFrame({
    'Year': [year],
    'Property Type': [property_type],
    'Negeri': [negeri],
    'Daerah': [daerah]
})

# Predict household price when the user clicks the "Predict" button
if st.button("Predict"):
    predicted_price = predict_household_price(user_input)
    st.success(f"Predicted Household Price: {predicted_price:.2f}")
