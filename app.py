import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os

from model_utils import CyberAttackModel, create_features

# Set page config
st.set_page_config(
    page_title="Cyber Attack Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data functions
@st.cache_data
def load_data():
    # Load analysis report
    with open('comprehensive_analysis_report.json', 'r') as f:
        report = json.load(f)
    
    # Load model results
    with open('model_results.json', 'r') as f:
        results = json.load(f)
        
    return report, results

@st.cache_data
def load_training_data():
    df = pd.read_csv('data/labelled_train.csv')
    return df

# Main App Structure
def main():
    st.title("🛡️ Cyber Attack Detection System")
    st.markdown("### Interactive Dashboard & Live Detection")

    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the App Mode",
        ["Dashboard", "Live Detection"])

    # Load data
    try:
        report, results = load_data()
    except FileNotFoundError:
        st.error("Data files not found. Please ensure 'comprehensive_analysis_report.json' and 'model_results.json' are in the directory.")
        return

    if app_mode == "Dashboard":
        dashboard_ui(report, results)
    elif app_mode == "Live Detection":
        live_detection_ui(report, results)

def dashboard_ui(report, results):
    # ... (Code unchanged, implicit keep)
    st.header("Project Overview")
    st.info(f"**Project Title:** {report['project_info']['title']}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Date:** {report['project_info']['date']}")
    with col2:
        st.write(f"**Dataset:** {report['project_info']['dataset']}")
    with col3:
        st.write(f"**Model Type:** {results['model_architecture']['type']}")

    st.divider()

    st.subheader("📊 Dataset Statistics")
    
    # Dataset Info
    ds_info = results['dataset_info']
    
    d_col1, d_col2, d_col3, d_col4 = st.columns(4)
    d_col1.metric("Train Samples", f"{ds_info['train_samples']:,}")
    d_col2.metric("Test Samples", f"{ds_info['test_samples']:,}")
    d_col3.metric("Malicious Samples (Test)", f"{ds_info['test_malicious_pct']:.2f}%")
    d_col4.metric("Class Imbalance Ratio", f"{ds_info['class_imbalance_ratio']:.2f}")

    st.divider()

    st.subheader("📈 Model Performance")
    
    metrics = results['performance_metrics']
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    m_col2.metric("Precision", f"{metrics['precision']:.4f}")
    m_col3.metric("Recall", f"{metrics['recall']:.4f}")
    m_col4.metric("F1 Score", f"{metrics['f1_score']:.4f}")
    
    st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = results['confusion_matrix']
    cm_data = pd.DataFrame(
        [[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]],
        index=["Actual Benign", "Actual Malicious"],
        columns=["Predicted Benign", "Predicted Malicious"]
    )
    st.table(cm_data)

    # Feature Importance
    st.subheader("🔍 Feature Importance")
    fi = results['feature_importance']
    fi_df = pd.DataFrame(list(fi.items()), columns=['Feature', 'Importance'])
    fi_df = fi_df.sort_values(by='Importance', ascending=True)
    
    fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h', title="Feature Importance Analysis")
    st.plotly_chart(fig, use_container_width=True)


def live_detection_ui(report, results):
    st.header("Live Threat Detection")
    st.write("This module allows you to train the neural network model and use it to predict whether a system call set is benign or malicious.")
    
    # Check session state for model
    if 'model' not in st.session_state:
        st.warning("⚠️ Model not trained. Please train the model to enable prediction.")
        
        if st.button("🚀 Train Model Now"):
            with st.spinner("Loading data and training Neural Network..."):
                # Load data
                df = load_training_data()
                feature_cols = ['processId', 'threadId', 'parentProcessId', 'userId', 'mountNamespace', 'argsNum', 'returnValue']
                target_col = 'sus_label'
                
                X = df[feature_cols].values.astype(np.float64)
                y = df[target_col].values.astype(np.float64)
                
                # Normalize
                mean = np.mean(X, axis=0)
                std = np.std(X, axis=0)
                std[std == 0] = 1
                X_scaled = (X - mean) / std
                
                # Enhance
                X_enh = create_features(X_scaled)
                
                # Train
                model = CyberAttackModel(input_dim=X_enh.shape[1])
                model.train(X_enh, y, epochs=30) # Use 30 epochs as per original script
                
                # Save to session
                st.session_state['model'] = model
                st.session_state['scaler_mean'] = mean
                st.session_state['scaler_std'] = std
                
            st.success("✅ Model trained successfully!")
            st.rerun()
            
    else:
        st.success("✅ Model is ready for inference.")
        if st.button("Reset / Retrain Model"):
            del st.session_state['model']
            st.rerun()
            
        st.subheader("🔮 Predict System Event")
        
        col1, col2 = st.columns(2)
        
        with col1:
            processId = st.number_input("Process ID", value=0)
            threadId = st.number_input("Thread ID", value=0)
            parentProcessId = st.number_input("Parent Process ID", value=0)
            userId = st.number_input("User ID", value=1000) # Default to common user
            
        with col2:
            mountNamespace = st.number_input("Mount Namespace", value=4026531840)
            argsNum = st.number_input("Number of Args", value=0)
            returnValue = st.number_input("Return Value", value=0)
            
        if st.button("Apply Prediction"):
            # Prepare input
            input_features = np.array([[processId, threadId, parentProcessId, userId, mountNamespace, argsNum, returnValue]], dtype=np.float64)
            
            # Scale
            mean = st.session_state['scaler_mean']
            std = st.session_state['scaler_std']
            input_scaled = (input_features - mean) / std
            
            # Enhance
            input_enh = create_features(input_scaled)
            
            # Predict
            model = st.session_state['model']
            prob = model.predict(input_enh)[0][0]
            
            st.divider()
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.metric("Malicious Probability", f"{prob:.4f}")
            
            with col_res2:
                if prob >= 0.5:
                    st.error("🚨 MALICIOUS EVENT DETECTED")
                else:
                    st.success("🛡️ BENIGN EVENT")


if __name__ == "__main__":
    main()
