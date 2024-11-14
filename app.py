import streamlit as st
import numpy as np
import pickle
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import base64
from io import BytesIO
import matplotlib.pyplot as plt

# Load the crop health model
model = load_model('f2.h5')

# List of states and commodities
states = [
    "Aandhra pradesh", "Bihar", "Chandigarh", "Chattisgarh", "Gujarat", "Haryana", 
    "Himachal pradesh", "Jammu and kashmir", "Karnataka", "Kerala", 
    "Madhya pradesh", "Maharashtra", "Meghalaya", "Delhi", 
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Telangana", 
    "Tripura", "Uttar pradesh", "Uttrakhand", "West bengal"
]

commodities = ["rice", "wheat", "potato", "tomato"]

# Create a mapping dictionary for each state
state_mapping = {state: i for i, state in enumerate(states)}

# Load the saved regression model for price prediction
with open('linear_regression_model.pkl', 'rb') as file:
    price_model = pickle.load(file)

# Function to convert image to base64
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Function to display price distribution pie chart
def show_price_distribution(min_price, modal_price, max_price):
    # Labels and data for pie chart
    labels = ['Minimum Price', 'Modal Price', 'Maximum Price']
    sizes = [min_price, modal_price, max_price]
    colors = ['#ff9999', '#41d81c', '#66b3ff']
    explode = (0.1, 0.1, 0)  # explode the first two slices for emphasis

    # Plotting the pie chart
    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.
    st.pyplot(fig)

# Function to preprocess the uploaded image for health prediction
def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (256, 256))  # Resize to 256x256 as per model's input size
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to make crop health predictions
def predict_health(image):
    preprocessed_img = preprocess_image(image)
    prediction = model.predict(preprocessed_img)
    return prediction

# Define a function to make price predictions
def predict_modal_price(inputs):
    inputs = np.array(inputs).reshape(1, -1)
    prediction = price_model.predict(inputs)
    return prediction[0]

# Streamlit UI
def main():
    st.title("🌾 Crop Prediction Dashboard")

    # Crop Health Detection
    st.header("🌱 Crop Health Detection")
    st.write("Upload an image of a crop leaf to detect its health status.")

    # File uploader for crop health image
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        class_names = [
            'Rice_healthy', 'Rice_unhealthy', 'Potato_healthy', 'Potato_unhealthy', 
            'Wheat_healthy', 'Wheat_unhealthy', 'Tomato_healthy', 'Tomato_unhealthy'
        ]
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image.", use_column_width=True)

            st.write("Classifying...")
            prediction = predict_health(image)
            predicted_class = class_names[np.argmax(prediction)]

            st.write(f"Predicted Class: {predicted_class}")

    # Crop Price Prediction
    st.header("💰 Crop Price Prediction")
    st.write("Enter details of the crop and market to predict its modal price.")

    # Input fields for price prediction
    state = st.selectbox("Select State", states)
    min_price = st.number_input("Minimum Wholesale Price per quintal (100 Kg)", min_value=0.0, max_value=50000.0, step=100.0)
    max_price = st.number_input("Maximum Wholesale Price per quintal (100 Kg)", min_value=0.0, max_value=50000.0, step=100.0)

    # Commodity selection
    commodity = st.radio("Select Commodity", commodities)

    # Convert selected commodity to a one-hot encoding
    commodity_values = [1 if commodity == c else 0 for c in commodities]

    # Convert categorical inputs to binary features based on mappings
    state_encoded = [1 if i == state_mapping[state] else 0 for i in range(len(states))]

    # Combine inputs for model
    input_data = [min_price, max_price] + commodity_values + state_encoded

    if st.button("Predict Modal Price"):
        modal_price = predict_modal_price(input_data)
        st.subheader(f"Predicted Wholesale Modal Price per quintal (100Kg): ₹{modal_price:.2f}")
    
        st.subheader("Price Distribution:")
     # Display the pie chart for price distribution
        show_price_distribution(min_price, modal_price, max_price)

# Run the Streamlit app
if __name__ == '__main__':
    main()
