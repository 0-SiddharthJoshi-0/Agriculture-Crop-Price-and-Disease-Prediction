import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import base64
from io import BytesIO
import time
import pickle

# Load the trained models
model = load_model('crop_health_model_V2.h5')

# Load the linear regression model
with open('linear_regression_model.pkl', 'rb') as file:
    linear_model = pickle.load(file)

# Function to convert image to base64
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # Save image to buffer
    return base64.b64encode(buffered.getvalue()).decode()  # Encode to base64

# Define a function to preprocess the uploaded image
def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (256, 256))  # Resize to 256x256 as per your model's input size
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Define a function to make predictions for crop health
def predict_health(image):
    preprocessed_img = preprocess_image(image)
    prediction = model.predict(preprocessed_img)
    return prediction

# Define a function to predict price using the linear regression model
def predict_price(features):
    features_array = np.array(features).reshape(1, -1)  # Reshape for single sample
    price_prediction = linear_model.predict(features_array)
    return price_prediction[0][0]  # Return the predicted price

# Define the Streamlit UI
def main():
    st.title("🌱 Crop Health Detection and Price Prediction Dashboard")
    
    # Crop Health Detection Section
    st.write("Upload an image of a crop leaf to detect its health status.")
    
    # File uploader for image upload
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.markdown(
                f"<div style='text-align: center;'><img src='data:image/png;base64,{image_to_base64(image)}' width='400' height='400'/></div>",
                unsafe_allow_html=True)

            st.write("Classifying...")

            # Predict the crop health
            prediction = predict_health(image)

            # Extract the prediction result (assuming binary classification: Healthy=0, Unhealthy=1)
            health_probability = prediction[0][1]  # probability for unhealthy class
            health_status = "Unhealthy" if health_probability > 0.5 else "Healthy"

            # Display the result
            st.subheader(f"The crop is {health_status}.")

            # Determine progress value
            health = 1 - health_probability
            progress_value = max(0.0, min(float(health), 1.0))  # Ensure it's a float between 0 and 1
            st.progress(progress_value)

            # Add a delay to simulate processing time
            time.sleep(1)  # Simulate a delay of 1 second

            # Display confidence levels
            st.write(f"Confidence (Healthy): {1 - health_probability:.2f}")
            st.write(f"Confidence (Unhealthy): {health_probability:.2f}")

            # Health recommendations
            if health_status == "Unhealthy":
                st.warning("Consider applying appropriate treatments or care.")
            else:
                st.success("The plant is healthy! 🌿")

    # Price Prediction Section
    st.write("Enter the following details for price prediction of commodities.")
    
    # Create input fields for the linear regression model
    min_price = st.number_input("Min Price", value=0)
    max_price = st.number_input("Max Price", value=0)
    commodity_potato = st.number_input("Commodity Potato (0 or 1)", value=0, min_value=0, max_value=1)
    commodity_rice = st.number_input("Commodity Rice (0 or 1)", value=0, min_value=0, max_value=1)
    commodity_tomato = st.number_input("Commodity Tomato (0 or 1)", value=0, min_value=0, max_value=1)
    commodity_wheat = st.number_input("Commodity Wheat (0 or 1)", value=0, min_value=0, max_value=1)
    grade_faq = st.number_input("Grade FAQ (0 or 1)", value=0, min_value=0, max_value=1)
    grade_medium = st.number_input("Grade Medium (0 or 1)", value=0, min_value=0, max_value=1)

    # Button to trigger prediction
    if st.button("Predict Price"):
        features = [min_price, max_price, commodity_potato, commodity_rice, commodity_tomato, commodity_wheat, grade_faq, grade_medium]
        predicted_price = predict_price(features)
        st.subheader(f"The predicted price for 100 kg of commodities is: {predicted_price:.2f}")

# Run the Streamlit app
if __name__ == '__main__':
    main()
