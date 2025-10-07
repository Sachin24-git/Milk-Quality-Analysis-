import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Milk Quality Analyzer",
    page_icon="🥛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .quality-premium { background-color: #d4edda; padding: 10px; border-radius: 5px; }
    .quality-standard { background-color: #fff3cd; padding: 10px; border-radius: 5px; }
    .quality-poor { background-color: #f8d7da; padding: 10px; border-radius: 5px; }
    .metric-card { 
        background-color: #f8f9fa; 
        padding: 15px; 
        border-radius: 10px; 
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class MilkQualityApp:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = ['pH', 'temperature', 'taste', 'odor', 'fat', 'turbidity', 'color']
        self.load_models()
    
    def load_models(self):
        """Load trained models"""
        try:
            self.model = joblib.load('models/kmeans_model.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            return True
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return False
    
    def predict_quality(self, input_data):
        """Predict milk quality"""
        try:
            # Scale input data
            input_scaled = self.scaler.transform([input_data])
            cluster = self.model.predict(input_scaled)[0]
            
            # Quality mapping
            quality_map = {
                0: {"label": "Premium Quality", "color": "green", "emoji": "✅"},
                1: {"label": "Standard Quality", "color": "blue", "emoji": "🔵"},
                2: {"label": "Average Quality", "color": "orange", "emoji": "🟠"},
                3: {"label": "Poor Quality", "color": "red", "emoji": "❌"},
                4: {"label": "Acidic - Needs Attention", "color": "darkred", "emoji": "🚨"}
            }
            
            return quality_map.get(cluster, {"label": "Unknown", "color": "gray", "emoji": "❓"})
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None
    
    def get_recommendations(self, quality_label, input_data):
        """Get recommendations based on quality"""
        recommendations = {
            "Premium Quality": [
                "✅ Excellent for premium dairy products",
                "✅ Ideal for direct consumption",
                "✅ Can be used for specialty products"
            ],
            "Standard Quality": [
                "✅ Suitable for standard dairy products",
                "✅ Good for pasteurization",
                "⚠️ Monitor storage conditions"
            ],
            "Average Quality": [
                "⚠️ Requires careful processing",
                "⚠️ Check storage temperature",
                "🔍 Conduct additional quality tests"
            ],
            "Poor Quality": [
                "❌ Consider rejection",
                "❌ Not suitable for consumption",
                "🔄 May be used for non-food purposes"
            ],
            "Acidic - Needs Attention": [
                "🚨 Immediate quality check required",
                "🚨 Investigate storage conditions",
                "🚨 Review handling procedures"
            ]
        }
        
        base_recommendations = recommendations.get(quality_label, ["No specific recommendations available."])
        
        # Add specific recommendations based on parameters
        if input_data[0] < 6.4:  # pH too low
            base_recommendations.append("🚨 pH level indicates potential spoilage")
        if input_data[1] > 6:  # Temperature too high
            base_recommendations.append("⚠️ Storage temperature is above recommended levels")
        if input_data[2] < 5:  # Taste score low
            base_recommendations.append("🔍 Low taste score - investigate processing")
        
        return base_recommendations

def main():
    # Header
    st.markdown('<h1 class="main-header">🥛 Milk Quality Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Initialize app
    app = MilkQualityApp()
    
    if not app.load_models():
        st.error("Please train the model first using data_training.py")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Analysis Mode",
        ["Single Sample Analysis", "Batch Analysis", "Quality Insights", "Data Overview"]
    )
    
    # Single Sample Analysis
    if app_mode == "Single Sample Analysis":
        st.header("🔍 Single Sample Quality Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Input Milk Parameters")
            
            # Create input fields
            with st.form("milk_quality_form"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    pH = st.slider("pH Level", 6.0, 7.5, 6.7, 0.1)
                    temperature = st.slider("Temperature (°C)", 0.0, 10.0, 4.0, 0.5)
                    taste = st.slider("Taste Score", 1, 10, 7)
                
                with col2:
                    odor = st.slider("Odor Score", 1, 10, 7)
                    fat = st.slider("Fat Content (%)", 2.0, 5.0, 3.5, 0.1)
                
                with col3:
                    turbidity = st.slider("Turbidity (NTU)", 0.5, 5.0, 2.5, 0.1)
                    color = st.slider("Color Score", 50, 100, 85)
                
                submitted = st.form_submit_button("Analyze Quality")
        
        if submitted:
            input_data = [pH, temperature, taste, odor, fat, turbidity, color]
            result = app.predict_quality(input_data)
            
            if result:
                with col2:
                    st.subheader("Quality Result")
                    
                    # Display quality result with color coding
                    quality_class = f"quality-{result['label'].split()[0].lower()}"
                    st.markdown(f"""
                    <div class="{quality_class}">
                        <h3>{result['emoji']} {result['label']}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display metrics
                    st.metric("pH Level", f"{pH:.2f}")
                    st.metric("Temperature", f"{temperature}°C")
                    st.metric("Taste Score", f"{taste}/10")
                    st.metric("Odor Score", f"{odor}/10")
                
                # Recommendations
                st.subheader("📋 Recommendations")
                recommendations = app.get_recommendations(result['label'], input_data)
                
                for rec in recommendations:
                    st.write(rec)
                
                # Quality gauge chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = taste,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Overall Quality Score"},
                    gauge = {
                        'axis': {'range': [None, 10]},
                        'bar': {'color': result['color']},
                        'steps': [
                            {'range': [0, 4], 'color': "lightgray"},
                            {'range': [4, 7], 'color': "yellow"},
                            {'range': [7, 10], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 6
                        }
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Batch Analysis
    elif app_mode == "Batch Analysis":
        st.header("📊 Batch Quality Analysis")
        
        uploaded_file = st.file_uploader("Upload CSV file with milk quality data", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Successfully uploaded {len(df)} samples")
                
                # Ensure all required columns are present
                missing_cols = [col for col in app.features if col not in df.columns]
                if missing_cols:
                    st.error(f"Missing columns: {', '.join(missing_cols)}")
                else:
                    # Predict quality for all samples
                    predictions = []
                    for _, row in df.iterrows():
                        input_data = [row[col] for col in app.features]
                        result = app.predict_quality(input_data)
                        predictions.append(result['label'] if result else "Unknown")
                    
                    df['Predicted_Quality'] = predictions
                    
                    # Display results
                    st.subheader("Batch Analysis Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        quality_counts = df['Predicted_Quality'].value_counts()
                        fig = px.pie(values=quality_counts.values, 
                                   names=quality_counts.index,
                                   title="Quality Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.scatter(df, x='pH', y='taste', color='Predicted_Quality',
                                       title="pH vs Taste by Quality")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col3:
                        st.dataframe(df[app.features + ['Predicted_Quality']].head(10))
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Analysis Results",
                        data=csv,
                        file_name="milk_quality_analysis.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    # Quality Insights
    elif app_mode == "Quality Insights":
        st.header("📈 Quality Insights & Trends")
        
        # Load sample data for demonstration
        try:
            df = pd.read_csv('data/milk_quality_clustered.csv')
            df['date'] = pd.to_datetime(df['date'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Quality trends over time
                st.subheader("Quality Trends Over Time")
                weekly_quality = df.groupby([pd.Grouper(key='date', freq='W'), 'cluster']).size().unstack(fill_value=0)
                fig = px.line(weekly_quality, title="Weekly Quality Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Feature correlations
                st.subheader("Feature Correlations")
                corr_matrix = df[app.features].corr()
                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto")
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistical summary
            st.subheader("Statistical Summary by Quality Cluster")
            cluster_summary = df.groupby('cluster')[app.features].mean().round(3)
            st.dataframe(cluster_summary)
            
        except Exception as e:
            st.warning("Sample data not available. Please run the training pipeline first.")
    
    # Data Overview
    else:
        st.header("📋 Data Overview & Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Information")
            if app.model:
                st.write(f"**Number of Clusters:** {app.model.n_clusters}")
                st.write(f"**Features Used:** {', '.join(app.features)}")
                st.write("**Cluster Centers:**")
                cluster_centers = pd.DataFrame(
                    app.model.cluster_centers_,
                    columns=app.features
                ).round(3)
                st.dataframe(cluster_centers)
        
        with col2:
            st.subheader("Quality Standards Reference")
            
            standards_data = {
                'Parameter': ['pH', 'Temperature', 'Taste', 'Odor', 'Fat'],
                'Premium Range': ['6.6-6.8', '2-4°C', '8-10', '8-10', '3.8-4.2%'],
                'Acceptable Range': ['6.4-6.9', '4-6°C', '6-8', '6-8', '3.2-4.0%'],
                'Critical Level': ['<6.3 or >7.0', '>8°C', '<5', '<5', '<2.8%']
            }
            
            standards_df = pd.DataFrame(standards_data)
            st.dataframe(standards_df)
        
        # Feature importance
        st.subheader("Feature Importance in Clustering")
        if app.model:
            importance = np.std(app.model.cluster_centers_, axis=0)
            fig = px.bar(x=app.features, y=importance, 
                        labels={'x': 'Features', 'y': 'Importance (Std Dev)'},
                        title="Feature Importance in Quality Clustering")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()