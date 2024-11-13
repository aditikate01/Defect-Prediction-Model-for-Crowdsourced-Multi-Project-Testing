import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import functions from other files
from src.data_loader import load_and_preprocess_data
from src.model import train_model
from src.utils import get_metrics_info

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Bug Prediction"])

# Main Page
if app_mode == "Home":
    st.header("SOFTWARE BUG PREDICTION SYSTEM")
    image_path = "buuuggg.jpg"  # Ensure this image is in the same directory as your script
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Software Bug Prediction System! 🐞🔍

    Our mission is to help identify potential bugs in your software projects efficiently. Upload your dataset, and our system will analyze it to detect any potential issues. Let's ensure a smoother development process together!

    ### How It Works
    1. **Upload Dataset:** Go to the **Bug Prediction** page and upload your dataset with software metrics.
    2. **Analysis:** Our system will process the dataset to identify potential bugs.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** We utilize advanced machine learning techniques for accurate bug detection.
    - **User-Friendly:** A simple and intuitive interface for a seamless user experience.
    - **Fast and Efficient:** Get results in seconds, enabling quick decision-making.

    ### Get Started
    Click on the **Bug Prediction** page in the sidebar to upload your dataset and experience the power of our Software Bug Prediction System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About the Project
    This system analyzes software metrics to predict potential bugs in your software projects. By utilizing machine learning, we aim to enhance software quality and reliability.

    #### Content
    - Upload datasets to identify potential bugs.
    - Get insights into software metrics and their correlation with bugs.
    """)

# Bug Prediction Page
elif app_mode == "Bug Prediction":
    st.header("Bug Prediction")
    
    # Load dataset with error handling
    try:
        bugs_csv_path = "bugs.csv"  # Path to your CSV file
        bugs_data = pd.read_csv(bugs_csv_path)
        st.write("CSV file loaded successfully.")
    except FileNotFoundError as e:
        st.error(f"File not found: {e.filename}. Please ensure the CSV file is in the correct directory.")
        st.stop()
    
    # Load and preprocess data
    result = load_and_preprocess_data(bugs_csv_path, 'Bugs')
    if result is not None:
        X_train, X_test, y_train, y_test, scaler, feature_names = result
        st.write("Data successfully loaded and preprocessed.")
    else:
        st.error("Failed to load and preprocess data. Please check the data file.")
        st.stop()
    
    # Train and store the model
    try:
        model = train_model(X_train, y_train)
        st.write("Model trained successfully.")
    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
        st.stop()
    
    # Display metrics information
    st.header('Bugs Data Information')
    metrics_df = get_metrics_info()
    st.dataframe(metrics_df)

    # Bug predictions
    st.header('Bug Prediction')
    st.write('Enter the software metrics to predict the likelihood of bugs:')
    
    input_data = {}
    for i, feature in enumerate(feature_names):
        input_data[feature] = st.number_input(
            f'Enter {feature}:', 
            value=int(X_train[:, list(feature_names).index(feature)].mean()), 
            key=f'input_{feature}_{i}'
        )

    if st.button('Predict'):
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict_proba(input_scaled)[0]
        st.write(f'Probability of having bugs: {prediction[1]:.2f}')
        st.write(f'Probability of not having bugs: {prediction[0]:.2f}')


# Load dataset with error handling
try:
    bugs_csv_path = "bugs.csv"  # Adjust this path as necessary
    bugs_data = pd.read_csv(bugs_csv_path, delimiter=';')  # Use the correct delimiter
    st.write("CSV file loaded successfully.")
except FileNotFoundError as e:
    st.error(f"File not found: {e.filename}. Please ensure the CSV file is in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the CSV file: {str(e)}")
    st.stop()

# Display the first few rows of the dataset to inspect data
st.write("First few rows of the dataset:")
st.dataframe(bugs_data.head())

# Ensure that all columns are numeric, coercing non-numeric to NaN
for col in bugs_data.columns:
    bugs_data[col] = pd.to_numeric(bugs_data[col], errors='coerce')

# Calculate the correlation matrix and drop non-numeric columns
numeric_data = bugs_data.select_dtypes(include='number')
bugs_corr = numeric_data.corr()

# Create and display the correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(bugs_corr, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
plt.title('Correlation Heatmap of Bugs Dataset')
st.pyplot(fig)

