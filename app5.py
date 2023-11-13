import numpy as np
import streamlit as st
import joblib
from PIL import Image

# Load the pre-trained model using joblib
with open('./best_model.pkl', 'rb') as model_file:
    model = joblib.load(model_file)

def preprocess_image(image):
    # Convert the image to grayscale
    image = image.convert('L')
    # Resize the image to 28x28 pixels
    image = image.resize((28, 28))
    # Convert the image to a numpy array and normalize pixel values
    image_array = np.array(image) / 255.0
    # Flatten the array to match the model input shape
    image_array = image_array.flatten()
    # Reshape the array to (1, 28, 28) as the model expects a batch dimension
    image_array = image_array.reshape(1, -1)
    return image_array

def predict_digit(image):
    # Preprocess the input image
    processed_image = preprocess_image(image)
    # Make a prediction using the pre-trained model
    prediction = model.predict(processed_image)
    # Get the predicted digit
    digit = np.argmax(prediction)
    return digit

def download_best_model():
    # Save the best model to a temporary file
    temp_file_path = "temp_best_model.pkl"
    joblib.dump(model, temp_file_path)

    # Create a download button for the best model file
    st.download_button(
        label="Download Best Model",
        data=open(temp_file_path, "rb").read(),
        file_name="best_model.pkl",
        key="download_best_model",
        help="Click to download the best model.",
    )

def main():
    st.title("MNIST Digit Recognition")
    st.write(
        "This app uses a pre-trained neural network to recognize digits from the MNIST dataset."
    )

    # Add a download button for the best model file
    download_best_model()

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        digit = predict_digit(image)

        st.success(f"The digit in the image is predicted to be: {digit}")

if __name__ == "__main__":
    main()
