import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import os
import requests
from pathlib import Path
import gdown

st.set_page_config(
    page_title="Detecção de Fadiga",
    layout="wide"
)

st.title("Sistema de Detecção de Fadiga")

st.sidebar.header("Configurações")

model_option = st.sidebar.selectbox(
    "Escolha o modelo:",
    ["CNN", "Transfer Learning"]
)

with st.sidebar.expander("Sobre os Modelos"):
    st.markdown("""
    **CNN:**
    - Modelo CNN customizado treinado
    - Tenta carregar: best_cnn_model.h5 → cnn_final.h5
    
    **Transfer Learning:**
    - Modelo com Transfer Learning
    - Tenta carregar: best_transfer_model.h5 → transfer_final.h5
    """)

threshold = st.sidebar.slider(
    "Threshold de Decisão",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Valores mais baixos aumentam sensibilidade para fadiga"
)

MODEL_IDS = {
    'best_cnn': '1uc1vLhyxv-kW2kYKj6Ul6uMzgU-ff4iH',
    'cnn_final': '1jz8SbiwkvlwYqgpdrcH-EHc4iwQ-bx1p', 
    'best_transfer': '176gsQwCJqiYjQ3ughK5xmsfuXVAmBhA5', 
    'transfer_final': '1W-TIRlkjBSUlFbjT4Z_Zq0d6GoC_8F2W'
}

def download_model_from_gdrive(file_id, output_path):
    """Baixa arquivo do Google Drive usando gdown"""
    if os.path.exists(output_path):
        return True
    
    try:
        st.info(f"Baixando modelo: {output_path}...")
        
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output_path, quiet=False)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            st.success(f"Modelo baixado com sucesso: {output_path}")
            return True
        else:
            st.error(f"Falha ao baixar {output_path}")
            return False
        
    except Exception as e:
        st.error(f"Erro ao baixar {output_path}: {str(e)}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

def create_cnn_model(input_shape=(128, 128, 3)):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def create_transfer_model(input_shape=(128, 128, 3)):
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=None
    )
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

@st.cache_resource
def load_model(model_type):
    try:
        if model_type == "CNN":
            best_path = 'best_cnn_model.h5'
            final_path = 'cnn_final.h5'
            
            if download_model_from_gdrive(MODEL_IDS['best_cnn'], best_path):
                model = create_cnn_model()
                model.load_weights(best_path)
                return model, "CNN (Best)", best_path
            elif download_model_from_gdrive(MODEL_IDS['cnn_final'], final_path):
                model = create_cnn_model()
                model.load_weights(final_path)
                return model, "CNN (Final)", final_path
            else:
                st.error("Nenhum modelo CNN encontrado ou baixado")
                return None, None, None
                
        else:
            best_path = 'best_transfer_model.h5'
            final_path = 'transfer_final.h5'
            
            if download_model_from_gdrive(MODEL_IDS['best_transfer'], best_path):
                model = create_transfer_model()
                model.load_weights(best_path)
                return model, "Transfer (Best)", best_path
            elif download_model_from_gdrive(MODEL_IDS['transfer_final'], final_path):
                model = create_transfer_model()
                model.load_weights(final_path)
                return model, "Transfer (Final)", final_path
            else:
                st.error("Nenhum modelo Transfer encontrado ou baixado")
                return None, None, None
                
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        import traceback
        st.error(f"Detalhes: {traceback.format_exc()}")
        return None, None, None

def preprocess_image(image, target_size=(128, 128)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize(target_size)
    
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def create_confidence_gauge(confidence, prediction):
    color = "red" if prediction == "Fatigue" else "green"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confiança da Predição", 'font': {'size': 20}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': 'lightgray'},
                {'range': [50, 75], 'color': 'gray'},
                {'range': [75, 100], 'color': 'darkgray'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_probability_chart(prob_fatigue):
    prob_non_fatigue = 1 - prob_fatigue
    
    fig = go.Figure(data=[
        go.Bar(
            x=['Fadiga', 'Não Fadiga'],
            y=[prob_fatigue * 100, prob_non_fatigue * 100],
            marker_color=['#FF6B6B', '#4ECDC4'],
            text=[f'{prob_fatigue*100:.1f}%', f'{prob_non_fatigue*100:.1f}%'],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Distribuição de Probabilidades",
        yaxis_title="Probabilidade (%)",
        xaxis_title="Classe",
        height=300,
        showlegend=False
    )
    
    return fig

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload da Imagem")
    uploaded_file = st.file_uploader(
        "Escolha uma imagem facial",
        type=['jpg', 'jpeg', 'png'],
        help="Formatos aceitos: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagem carregada", use_container_width=True)
        
        st.info(f"Dimensões: {image.size[0]}x{image.size[1]} | Formato: {image.format}")

with col2:
    st.subheader("Resultados da Análise")
    
    if uploaded_file is not None:
        with st.spinner("Carregando modelo..."):
            model, model_name, model_path = load_model(model_option)
        
        if model is not None:
            with st.spinner("Analisando imagem..."):
                processed_img = preprocess_image(image)
                prediction_prob = model.predict(processed_img, verbose=0)[0][0]
                
                predicted_class = "NonFatigue" if prediction_prob > threshold else "Fatigue"
                confidence = prediction_prob if prediction_prob > threshold else 1 - prediction_prob
                
                if predicted_class == "Fatigue":
                    st.error("FADIGA DETECTADA")
                    st.markdown("### A pessoa apresenta sinais de fadiga")
                else:
                    st.success("SEM FADIGA")
                    st.markdown("### A pessoa está alerta")
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("Modelo", model_name)
                with metric_col2:
                    st.metric("Confiança", f"{confidence*100:.1f}%")
                with metric_col3:
                    st.metric("Threshold", f"{threshold*100:.0f}%")
                
                st.caption(f"Arquivo: {model_path}")

if uploaded_file is not None:
    if 'model' in locals() and model is not None:
        st.markdown("---")
        st.subheader("Visualizações Detalhadas")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            gauge_fig = create_confidence_gauge(confidence, predicted_class)
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        with viz_col2:
            prob_fig = create_probability_chart(prediction_prob)
            st.plotly_chart(prob_fig, use_container_width=True)
