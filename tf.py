import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the TFLite model
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Define class names
class_names = [
    'Background_without_leaves', 'Eggplant_Aphids', 'Eggplant_Cercospora Leaf Spot',
    'Eggplant_Defect', 'Eggplant_Flea Beetles', 'Eggplant_Fresh', 'Eggplant_Fresh_Leaf',
    'Eggplant_Leaf Wilt', 'Eggplant_Phytophthora Blight', 'Eggplant_Powdery Mildew',
    'Eggplant_Tobacco Mosaic Virus', 'Okra_Alternaria Leaf Spot', 'Okra_Cercospora Leaf Spot',
    'Okra_Downy Mildew', 'Okra_Healthy', 'Okra_Leaf curly virus', 'Okra_Phyllosticta leaf spot',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_healthy', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_mosaic_virus', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites Two-spotted_spider_mite', 'Tomato_Target_Spot',
    'Tomato_Yellow_Leaf_Curl_Virus'
]

# Load the TFLite model
model_path = r"C:\Users\shyam\Downloads\plant_Project_Vidhya\Project_Bobby\model.tflite"
interpreter = load_tflite_model(model_path)

# Function to preprocess and predict
def predict_image_tflite(image_file):
    img = Image.open(image_file)
    img = img.convert("RGB")
    img = img.resize((160, 160))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Get model input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Get predicted class and confidence
    pred_class = class_names[np.argmax(output_data[0])]
    pred_confidence = f"{round(100 * np.max(output_data[0]), 2)}%"

    return pred_class, pred_confidence

# Streamlit UI
st.title("Plant Disease Detection (TFLite Model)")
st.write("Upload an image to classify plant diseases.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            result, confidence = predict_image_tflite(uploaded_file)
        st.success(f"Prediction: {result}")
        st.info(f"Confidence: {confidence}")

# Run the app using: streamlit run app_streamlit.py