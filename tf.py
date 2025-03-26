import os
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Set Page Configuration
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="centered")

# Hide Streamlit branding
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stButton>button {
            width: 100%;
            font-size: 18px;
            padding: 10px;
        }
        .stImage img {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
        }
        .stSuccess, .stWarning, .stInfo {
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
@st.cache_resource
def load_tflite_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found at {model_path}")

    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load model: {e}")

model_path = "model.tflite"  # Ensure correct relative path
interpreter = load_tflite_model(model_path)


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

# Disease resolutions with fertilizers and pesticides
disease_resolutions = {
        "Eggplant_Leaf Wilt": {
        "Fertilizer": "Seaweed Extract",
        "NPK": "5-5-10",
        "Pesticide": "Copper-based fungicide",
        "Tips": ["Ensure proper drainage.", "Avoid overwatering.", "Apply mulch to retain soil moisture."]
    },
    "Eggplant_Phytophthora Blight": {
        "Fertilizer": "Composted Manure",
        "NPK": "10-5-10",
        "Pesticide": "Metalaxyl-based fungicide",
        "Tips": ["Improve soil drainage.", "Rotate crops annually.", "Remove infected plants immediately."]
    },
    "Eggplant_Tobacco Mosaic Virus": {
        "Fertilizer": "Potassium Sulfate",
        "NPK": "8-16-16",
        "Pesticide": "No chemical cure; use resistant varieties",
        "Tips": ["Remove infected leaves.", "Disinfect gardening tools.", "Avoid handling plants after smoking."]
    },
    "Okra_Alternaria Leaf Spot": {
        "Fertilizer": "Fish Emulsion",
        "NPK": "5-1-1",
        "Pesticide": "Chlorothalonil-based fungicide",
        "Tips": ["Avoid overhead watering.", "Increase spacing between plants.", "Remove affected leaves promptly."]
    },
    "Okra_Cercospora Leaf Spot": {
        "Fertilizer": "Bone Meal",
        "NPK": "4-12-0",
        "Pesticide": "Copper-based fungicide",
        "Tips": ["Avoid wetting leaves.", "Apply mulch around plants.", "Rotate crops regularly."]
    },
    "Okra_Downy Mildew": {
        "Fertilizer": "Liquid Seaweed",
        "NPK": "2-3-1",
        "Pesticide": "Mancozeb-based fungicide",
        "Tips": ["Increase air circulation.", "Apply fungicide at early stages.", "Use well-draining soil."]
    },
    "Okra_Healthy": {
        "Fertilizer": "Organic Compost",
        "NPK": "10-10-10",
        "Pesticide": "None required",
        "Tips": ["Maintain regular watering.", "Ensure good sunlight exposure.", "Use organic fertilizers."]
    },
    "Okra_Leaf curly virus": {
        "Fertilizer": "Balanced Liquid Fertilizer",
        "NPK": "20-20-20",
        "Pesticide": "No cure; use resistant varieties",
        "Tips": ["Control whiteflies as they spread the virus.", "Remove infected plants.", "Use reflective mulch."]
    },
    "Okra_Phyllosticta leaf spot": {
        "Fertilizer": "Phosphorus-Rich Fertilizer",
        "NPK": "15-30-15",
        "Pesticide": "Copper-based fungicide",
        "Tips": ["Avoid excessive nitrogen fertilizers.", "Increase spacing between plants.", "Remove diseased leaves."]
    },
    "Tomato_Early_blight": {
        "Fertilizer": "Compost Tea",
        "NPK": "4-6-8",
        "Pesticide": "Chlorothalonil or Copper fungicide",
        "Tips": ["Prune lower leaves.", "Mulch soil to reduce spore spread.", "Water at the base."]
    },
    "Tomato_healthy": {
        "Fertilizer": "Organic Compost",
        "NPK": "10-10-10",
        "Pesticide": "None required",
        "Tips": ["Provide proper sunlight.", "Ensure adequate watering.", "Support plants with stakes or cages."]
    },
    "Tomato_Late_blight": {
        "Fertilizer": "Potassium-Rich Fertilizer",
        "NPK": "8-16-16",
        "Pesticide": "Mancozeb or Copper fungicide",
        "Tips": ["Remove infected plants immediately.", "Use resistant tomato varieties.", "Avoid excessive moisture."]
    },
    "Tomato_Leaf_Mold": {
        "Fertilizer": "Nitrogen-Rich Fertilizer",
        "NPK": "12-6-6",
        "Pesticide": "Sulfur-based fungicide",
        "Tips": ["Improve air circulation.", "Reduce humidity levels.", "Prune dense foliage."]
    },
    "Tomato_mosaic_virus": {
        "Fertilizer": "Balanced Organic Fertilizer",
        "NPK": "10-10-10",
        "Pesticide": "No chemical cure; control aphids",
        "Tips": ["Avoid handling infected plants.", "Disinfect tools regularly.", "Use virus-free seeds."]
    },
    "Tomato_Septoria_leaf_spot": {
        "Fertilizer": "Phosphorus-Enriched Fertilizer",
        "NPK": "15-30-15",
        "Pesticide": "Copper-based fungicide",
        "Tips": ["Remove lower infected leaves.", "Avoid overhead watering.", "Apply mulch to reduce splash."]
    },
    "Tomato_Spider_mites Two-spotted_spider_mite": {
        "Fertilizer": "Compost Tea",
        "NPK": "4-6-8",
        "Pesticide": "Neem oil or insecticidal soap",
        "Tips": ["Spray plants with water.", "Introduce predatory mites.", "Maintain high humidity."]
    },
    "Tomato_Target_Spot": {
        "Fertilizer": "Nitrogen-Rich Fertilizer",
        "NPK": "14-7-14",
        "Pesticide": "Chlorothalonil-based fungicide",
        "Tips": ["Improve soil drainage.", "Remove infected debris.", "Ensure good air circulation."]
    },
    "Tomato_Yellow_Leaf_Curl_Virus": {
        "Fertilizer": "Balanced Liquid Fertilizer",
        "NPK": "20-20-20",
        "Pesticide": "No chemical cure; control whiteflies",
        "Tips": ["Use reflective mulch.", "Plant resistant varieties.", "Control whitefly populations."]
    },
        "Eggplant_Fresh": {
        "Fertilizer": "Organic Compost",
        "NPK": "10-10-10",
        "Pesticide": "None required",
        "Tips": ["Ensure proper watering.", "Provide adequate sunlight.", "Use organic fertilizers."]
    },
    "Eggplant_Fresh_Leaf": {
        "Fertilizer": "Balanced Liquid Fertilizer",
        "NPK": "20-20-20",
        "Pesticide": "None required",
        "Tips": ["Maintain healthy soil conditions.", "Avoid overwatering.", "Prune regularly to promote growth."]
    },
    "Eggplant_Defect": {
        "Fertilizer": "Potassium-Enriched Fertilizer",
        "NPK": "8-16-16",
        "Pesticide": "Neem oil or organic pesticide",
        "Tips": ["Remove affected parts.", "Monitor for pests.", "Maintain proper nutrition."]
    },
    "Tomato_Bacterial_spot": {
        "Fertilizer": "Calcium Nitrate",
        "NPK": "15-5-30",
        "Pesticide": "Copper-based bactericide",
        "Tips": ["Avoid overhead watering.", "Remove infected leaves.", "Use disease-resistant varieties."]
    },
    "Eggplant_Cercospora Leaf Spot": {
        "Fertilizer": "Bone Meal",
        "NPK": "4-12-0",
        "Pesticide": "Copper-based fungicide",
        "Tips": ["Water at the base of plants.", "Remove affected leaves.", "Increase spacing for air circulation."]
    },
    "Eggplant_Aphids": {
        "Fertilizer": "Vermicompost",
        "NPK": "10-10-10",
        "Pesticide": "Neem oil or Imidacloprid",
        "Tips": ["Introduce ladybugs.", "Spray water on leaves.", "Avoid over-fertilizing."]
    },
    "Okra_Healthy": {
        "Fertilizer": "Organic Compost",
        "NPK": "10-10-10",
        "Pesticide": "None required",
        "Tips": ["Maintain regular watering.", "Ensure good sunlight exposure.", "Use organic fertilizers."]
    },
    "Okra_Leaf curly virus": {
        "Fertilizer": "Balanced Liquid Fertilizer",
        "NPK": "20-20-20",
        "Pesticide": "No cure; use resistant varieties",
        "Tips": ["Control whiteflies as they spread the virus.", "Remove infected plants.", "Use reflective mulch."]
    },
    "Okra_Phyllosticta leaf spot": {
        "Fertilizer": "Phosphorus-Rich Fertilizer",
        "NPK": "15-30-15",
        "Pesticide": "Copper-based fungicide",
        "Tips": ["Avoid excessive nitrogen fertilizers.", "Increase spacing between plants.", "Remove diseased leaves."]
    },
    "Tomato_Target_Spot": {
        "Fertilizer": "Nitrogen-Rich Fertilizer",
        "NPK": "14-7-14",
        "Pesticide": "Chlorothalonil-based fungicide",
        "Tips": ["Improve soil drainage.", "Remove infected debris.", "Ensure good air circulation."]
    },
    "Eggplant_Flea_Beetles": {
        "Fertilizer": "Balanced Organic Fertilizer",
        "NPK": "10-10-10",
        "Pesticide": "Neem oil or Pyrethrin-based insecticide",
        "Tips": ["Use floating row covers to prevent infestation.", "Encourage natural predators like ladybugs.", "Apply neem oil regularly."]
    },
    "Eggplant_Powdery Mildew": {
        "Fertilizer": "Sulfur-based Fertilizer",
        "NPK": "5-10-5",
        "Pesticide": "Sulfur or Potassium bicarbonate-based fungicide",
        "Tips": ["Improve air circulation around plants.", "Water plants at the base to keep foliage dry.", "Remove infected leaves promptly."]
    },
    "Tomato_Spider_mites Two-spotted_spider_mite": {
        "Fertilizer": "Compost Tea",
        "NPK": "4-6-8",
        "Pesticide": "Neem oil or insecticidal soap",
        "Tips": ["Spray plants with water to dislodge mites.", "Introduce predatory mites as biological control.", "Maintain high humidity to deter spider mites."]
    }
}

def predict_image_tflite(image_file):
    img = Image.open(image_file).convert("RGB").resize((160, 160))
    img_array = np.array(img, dtype=np.float32)  # Do not normalize
    img_array = np.expand_dims(img_array, axis=0)  # Ensure batch dimension

    # Get model input/output details
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    interpreter.set_tensor(input_index, img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_index)

    # Get predicted class and confidence
    pred_index = np.argmax(output_data[0])
    pred_class = class_names[pred_index]
    pred_confidence = f"{round(100 * np.max(output_data[0]), 2)}%"

    return pred_class, pred_confidence




# Streamlit UI
st.title("🌿 Plant Disease Detection")
st.write("📸 Upload an image to classify plant diseases.")

# File uploader
uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("🔄 Analyzing..."):
            result, confidence = predict_image_tflite(uploaded_file)
        
        st.success(f"✅ Prediction: {result}")
        st.info(f"📊 Confidence: {confidence}")

        # Show Disease Resolution if available
        st.subheader("🩺 Treatment & Care")
        res = disease_resolutions.get(result)

        if res:
            st.write(f"🌱 **Recommended Fertilizer:** {res['Fertilizer']}")
            st.write(f"🧪 **NPK Ratio:** {res['NPK']}")
            st.write(f"🦠 **Recommended Pesticide:** {res['Pesticide']}")
            st.subheader("📌 Additional Tips:")
            for tip in res["Tips"]:
                st.write(f"- {tip}")
        else:
            st.warning("⚠️ No treatment information available for this disease.")

    # Button to restart
    if st.button("🔄 Try Another Image"):
        st.rerun()
