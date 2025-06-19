import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stroke Prediction App",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .low-risk {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model and preprocessing objects
@st.cache_resource
def load_model_and_preprocessors():
    try:
        # Load the trained model
        with open('stroke_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load the scaler
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            
        # Load label encoders
        with open('label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
            
        return model, scaler, label_encoders
    except FileNotFoundError:
        st.error("Model files not found. Please ensure all model files are uploaded.")
        return None, None, None

# Main title
st.markdown('<h1 class="main-header">🏥 Stroke Prediction Application</h1>', unsafe_allow_html=True)

# Load model and preprocessors
model, scaler, label_encoders = load_model_and_preprocessors()

if model is None:
    st.stop()

# Sidebar for input features
st.sidebar.markdown('<h2 class="sub-header">Patient Information</h2>', unsafe_allow_html=True)

# Input fields
col1, col2 = st.columns(2)

with col1:
    st.markdown('<h3 class="sub-header">Basic Information</h3>', unsafe_allow_html=True)
    
    gender = st.selectbox(
        "Gender",
        options=['Male', 'Female'],
        help="Patient's gender"
    )
    
    age = st.slider(
        "Age",
        min_value=0,
        max_value=100,
        value=50,
        help="Patient's age in years"
    )
    
    ever_married = st.selectbox(
        "Marital Status",
        options=['Yes', 'No'],
        help="Has the patient ever been married?"
    )
    
    residence_type = st.selectbox(
        "Residence Type",
        options=['Urban', 'Rural'],
        help="Type of residence area"
    )

with col2:
    st.markdown('<h3 class="sub-header">Health Information</h3>', unsafe_allow_html=True)
    
    work_type = st.selectbox(
        "Work Type",
        options=['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'],
        help="Type of work the patient does"
    )
    
    avg_glucose_level = st.number_input(
        "Average Glucose Level (mg/dL)",
        min_value=50.0,
        max_value=300.0,
        value=100.0,
        step=0.1,
        help="Average glucose level in blood"
    )
    
    bmi = st.number_input(
        "BMI (Body Mass Index)",
        min_value=10.0,
        max_value=50.0,
        value=25.0,
        step=0.1,
        help="Body Mass Index"
    )
    
    smoking_status = st.selectbox(
        "Smoking Status",
        options=['never smoked', 'formerly smoked', 'smokes', 'Unknown'],
        help="Patient's smoking history"
    )

# Medical history checkboxes
st.markdown('<h3 class="sub-header">Medical History</h3>', unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:
    hypertension = st.checkbox(
        "Hypertension",
        help="Does the patient have hypertension?"
    )

with col4:
    heart_disease = st.checkbox(
        "Heart Disease",
        help="Does the patient have heart disease?"
    )

# Prediction button
if st.button("🔍 Predict Stroke Risk", type="primary"):
    # Prepare input data
    input_data = {
        'gender': gender,
        'age': age,
        'hypertension': int(hypertension),
        'heart_disease': int(heart_disease),
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': Residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }
    
    # Create DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Feature Engineering - Create GlucoseBMI_Interaction
    input_df['GlucoseBMI_Interaction'] = input_df['avg_glucose_level'] * input_df['bmi']
    
    # Apply label encoding
    categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    
    for col in categorical_cols:
        if col in label_encoders:
            # Handle unseen categories
            try:
                input_df[col] = label_encoders[col].transform(input_df[col])
            except ValueError:
                # If category not seen during training, use the most frequent category
                input_df[col] = 0  # Default encoding
    
    # Scale numerical features
    numerical_cols = ['age', 'avg_glucose_level', 'bmi', 'GlucoseBMI_Interaction']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]
    
    # Display results
    st.markdown("---")
    st.markdown('<h2 class="sub-header">Prediction Results</h2>', unsafe_allow_html=True)
    
    col5, col6 = st.columns(2)
    
    with col5:
        if prediction == 1:
            st.markdown(
                f'<div class="prediction-box high-risk">'
                f'<h3>⚠️ HIGH RISK</h3>'
                f'<p>The model predicts a <strong>HIGH RISK</strong> of stroke for this patient.</p>'
                f'<p><strong>Confidence:</strong> {prediction_proba[1]:.2%}</p>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="prediction-box low-risk">'
                f'<h3>✅ LOW RISK</h3>'
                f'<p>The model predicts a <strong>LOW RISK</strong> of stroke for this patient.</p>'
                f'<p><strong>Confidence:</strong> {prediction_proba[0]:.2%}</p>'
                f'</div>',
                unsafe_allow_html=True
            )
    
    with col6:
        st.markdown("### Risk Probability")
        
        # Create a simple probability display
        risk_percentage = prediction_proba[1] * 100
        
        st.metric(
            label="Stroke Risk Probability",
            value=f"{risk_percentage:.1f}%",
            delta=None
        )
        
        # Risk interpretation
        if risk_percentage < 30:
            risk_level = "Low Risk"
            color = "green"
        elif risk_percentage < 70:
            risk_level = "Moderate Risk"
            color = "orange"
        else:
            risk_level = "High Risk"
            color = "red"
        
        st.markdown(f"**Risk Level:** <span style='color: {color}'>{risk_level}</span>", unsafe_allow_html=True)

# Model information
st.markdown("---")
with st.expander("ℹ️ About This Model"):
    st.markdown("""
    ### Model Information
    
    This stroke prediction model was trained using machine learning techniques on healthcare data. 
    The model uses the following features to make predictions:
    
    - **Demographics**: Age, Gender, Marital Status, Residence Type
    - **Work Information**: Work Type
    - **Health Metrics**: Average Glucose Level, BMI
    - **Medical History**: Hypertension, Heart Disease
    - **Lifestyle**: Smoking Status
    - **Engineered Feature**: Glucose-BMI Interaction
    
    ### Model Performance
    - **Algorithm**: Logistic Regression (Tuned)
    - **ROC AUC Score**: 0.83
    - **Key Strength**: Good at distinguishing between stroke and non-stroke cases
    
    ### Important Disclaimer
    ⚠️ **This tool is for educational purposes only and should NOT be used as a substitute for professional medical advice. 
    Always consult with healthcare professionals for medical decisions.**
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Developed for Educational Purposes | Stroke Prediction ML Model"
    "</div>",
    unsafe_allow_html=True
)