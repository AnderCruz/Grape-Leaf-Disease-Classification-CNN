import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
import io
import gdown
import plotly.express as px
import os

@st.cache_resource
def load_model():
    model_path = 'model_fp16_fast.tflite'
    
    # Check if file already exists, if not, download it
    if not os.path.exists(model_path):
        st.info("📥 Downloading model...")
        url = 'https://drive.google.com/uc?id=1Wz8wIWrdcCAsIsoewJ705Qn6Tn3OopOT'
        try:
            gdown.download(url, model_path, quiet=False)
            st.success("✅ Model download completed!")
        except Exception as e:
            st.error(f"❌ Download error: {str(e)}")
            return None
    
    # Check if file was downloaded correctly
    if not os.path.exists(model_path):
        st.error("❌ Model file was not downloaded correctly")
        return None
    
    try:
        # Load the model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        st.success("✅ Model loaded successfully!")
        
        # Show model information
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        st.sidebar.info(f"**Input shape:** {input_details[0]['shape']}")
        st.sidebar.info(f"**Output shape:** {output_details[0]['shape']}")
        
        return interpreter
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def load_image(interpreter):
    st.subheader("📷 Image Upload")
    uploaded_file = st.file_uploader(
        "Drag and drop an image here or click to select one", 
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_file is not None:
        try:
            # Read the image
            image_data = uploaded_file.read()
            image = Image.open(io.BytesIO(image_data))

            # Show the loaded image
            st.image(image, caption="Loaded Image", use_container_width=True)
            st.success("✅ Image loaded successfully!")

            # Get expected size from model automatically
            input_details = interpreter.get_input_details()
            expected_height = input_details[0]['shape'][1]
            expected_width = input_details[0]['shape'][2]
            
            st.info(f"📐 Resizing image to {expected_width}x{expected_height} pixels")
            
            # Resize to model's expected size
            image = image.resize((expected_width, expected_height))
            image = np.array(image, dtype=np.float32)
            
            # Check if image has 3 channels (RGB)
            if len(image.shape) == 2:  # If grayscale
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[-1] == 4:  # If has alpha channel (RGBA)
                image = image[..., :3]
            
            image = image / 255.0  # Normalization to range [0, 1]
            image = np.expand_dims(image, axis=0)

            st.success(f"✅ Image processed: {image.shape}")
            return image
            
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            return None
    
    return None

def prediction(interpreter, image):
    st.subheader("🔍 Analysis Results")
    
    try:
        # Get input and output tensor details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Check if image shape matches model's expected shape
        expected_shape = input_details[0]['shape']
        if image.shape != tuple(expected_shape):
            st.error(f"❌ Incompatible shape: Image {image.shape} vs Model {expected_shape}")
            return

        # Set input tensor for the model
        interpreter.set_tensor(input_details[0]['index'], image)

        # Run inference
        interpreter.invoke()

        # Get model output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Define classes
        classes = ['BlackMeasles', 'BlackRot', 'HealthyGrapes', 'LeafBlight']
        
        # Create DataFrame with results
        df = pd.DataFrame({
            'Disease': classes,
            'Probability (%)': 100 * output_data[0]
        })
        
        # Sort by probability
        df = df.sort_values('Probability (%)', ascending=False)
        
        # Show main result
        main_disease = df.iloc[0]
        st.metric(
            label="🔬 Main Diagnosis", 
            value=main_disease['Disease'],
            delta=f"{main_disease['Probability (%)']:.1f}%"
        )
        
        # Create chart
        fig = px.bar(
            df, 
            y='Disease', 
            x='Probability (%)', 
            orientation='h', 
            text='Probability (%)',
            title='Probability Distribution by Disease',
            color='Probability (%)',
            color_continuous_scale='viridis'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        
        st.plotly_chart(fig)
        
        # Show results table
        st.subheader("📊 Results Table")
        st.dataframe(df.style.format({'Probability (%)': '{:.2f}%'}))
        
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")

def main():
    st.set_page_config(
        page_title="Grape Vine Disease Classifier",
        page_icon="🍇",
        layout="wide"
    )
    
    st.title("🍇 Grape Vine Leaf Disease Classifier")
    st.markdown("---")
    
    # Sidebar with information
    st.sidebar.title("ℹ️ Information")
    st.sidebar.info(
        "This application uses artificial intelligence to identify diseases "
        "in grape vine leaves. Upload an image for analysis."
    )
    
    # Load model
    with st.spinner('🚀 Initializing model...'):
        interpreter = load_model()
    
    if interpreter is None:
        st.error("Could not load the model. Please try reloading the page.")
        return
    
    # Load image (now passing interpreter as parameter)
    image = load_image(interpreter)
    
    # Make prediction if image was loaded
    if image is not None:
        with st.spinner('🔍 Analyzing image...'):
            prediction(interpreter, image)

if __name__ == "__main__":
    main()

