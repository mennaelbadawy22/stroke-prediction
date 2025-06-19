import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
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

# Pre-trained model (simplified version for demonstration)
@st.cache_resource
def get_pretrained_model():
    """Return a pre-configured model with typical coefficients for stroke prediction"""
    
    # Create a logistic regression model with coefficients based on domain knowledge
    model = LogisticRegression()
    
    # These coefficients are based on typical stroke risk factors
    # Age, hypertension, heart_disease, avg_glucose_level, bmi have positive correlation with stroke
    # Gender, ever_married, work_type, residence_type, smoking_status vary
    model.coef_ = np.array([[
        0.05,   # age - positive correlation
        0.1,    # gender - slight effect
        1.2,    # hypertension - strong positive
        1.5,    # heart_disease - strong positive  
        0.3,    # ever_married - slight positive
        0.1,    # work_type - minimal effect
        0.05,   # residence_type - minimal effect
        0.8,    # avg_glucose_level - positive
        0.6,    # bmi - positive
        0.4,    # smoking_status - moderate positive
        0.7     # GlucoseBMI_Interaction - positive
    ]])
    
    model.intercept_ = np.array([-8.0])  # Negative intercept since stroke is rare
    model.classes_ = np.array([0, 1])
    
    return model

def get_scaler():
    """Return a pre-configured scaler"""
    scaler = StandardScaler()
    # Set mean and scale based on typical health data ranges
    scaler.mean_ = np.array([50, 0.5, 0.1, 0.1, 0.6, 2.0, 0.5, 100, 25, 1.5, 2500])
    scaler.scale_ = np.array([20, 0.5, 0.3, 0.3, 0.5, 1.5, 0.5, 40, 7, 1.2, 1000])
    return scaler

def get_label_encoders():
    """Return pre-configured label encoders"""
    label_encoders = {
        'gender': {'Female': 0, 'Male': 1},
        'ever_married': {'No': 0, 'Yes': 1},
        'work_type': {'Govt_job': 0, 'Never_worked': 1, 'Private': 2, 'Self-employed': 3, 'children': 4},
        'Residence_type': {'Rural': 0, 'Urban': 1},
        'smoking_status': {'Unknown': 0, 'formerly smoked': 1, 'never smoked': 2, 'smokes': 3}
    }
    return label_encoders

# Load model and preprocessors
model = get_pretrained_model()
scaler = get_scaler()
label_encoders = get_label_encoders()

# Main title
st.markdown('<h1 class="main-header">🏥 Stroke Prediction Application</h1>', unsafe_allow_html=True)

st.info("🔬 **Demo Version**: This app uses a pre-configured model for demonstration purposes. For production use, train with your own dataset.")

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
        'age': age,
        'gender': label_encoders['gender'][gender],
        'hypertension': int(hypertension),
        'heart_disease': int(heart_disease),
        'ever_married': label_encoders['ever_married'][ever_married],
        'work_type': label_encoders['work_type'][work_type],
        'Residence_type': label_encoders['Residence_type'][residence_type],
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': label_encoders['smoking_status'][smoking_status],
        'GlucoseBMI_Interaction': avg_glucose_level * bmi
    }
    
    # Create feature array
    features = np.array([list(input_data.values())])
    
    # Scale features (simplified scaling)
    features_scaled = (features - scaler.mean_) / scaler.scale_
    
    # Make prediction
    prediction_proba = 1 / (1 + np.exp(-(np.dot(features_scaled, model.coef_.T) + model.intercept_)))
    prediction = (prediction_proba > 0.5).astype(int)
    
    # Display results
    st.markdown("---")
    st.markdown('<h2 class="sub-header">Prediction Results</h2>', unsafe_allow_html=True)
    
    col5, col6 = st.columns(2)
    
    with col5:
        if prediction[0][0] == 1:
            st.markdown(
                f'<div class="prediction-box high-risk">'
                f'<h3>⚠️ HIGH RISK</h3>'
                f'<p>The model predicts a <strong>HIGH RISK</strong> of stroke for this patient.</p>'
                f'<p><strong>Confidence:</strong> {prediction_proba[0][0]:.2%}</p>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="prediction-box low-risk">'
                f'<h3>✅ LOW RISK</h3>'
                f'<p>The model predicts a <strong>LOW RISK</strong> of stroke for this patient.</p>'
                f'<p><strong>Confidence:</strong> {1-prediction_proba[0][0]:.2%}</p>'
                f'</div>',
                unsafe_allow_html=True
            )
    
    with col6:
        st.markdown("### Risk Probability")
        
        # Create a simple probability display
        risk_percentage = prediction_proba[0][0] * 100
        
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
    
    This stroke prediction model uses logistic regression to predict stroke risk based on various patient factors:
    
    - **Demographics**: Age, Gender, Marital Status, Residence Type
    - **Work Information**: Work Type
    - **Health Metrics**: Average Glucose Level, BMI
    - **Medical History**: Hypertension, Heart Disease
    - **Lifestyle**: Smoking Status
    - **Engineered Feature**: Glucose-BMI Interaction
    
    ### Model Details
    - **Algorithm**: Logistic Regression
    - **Features**: 11 input features (10 original + 1 engineered)
    - **Output**: Binary classification (High Risk / Low Risk) with probability score
    
    ### Important Disclaimer
    ⚠️ **This is a demonstration version with a pre-configured model. 
    This tool is for educational purposes only and should NOT be used as a substitute for professional medical advice. 
    Always consult with healthcare professionals for medical decisions.**
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Developed for Educational Purposes | Stroke Prediction ML Model Demo"
    "</div>",
    unsafe_allow_html=True
)