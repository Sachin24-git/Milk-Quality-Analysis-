import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class MilkQualityTester:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = ['pH', 'temperature', 'taste', 'odor', 'fat', 'turbidity', 'color']
    
    def load_models(self):
        """Load trained model and scaler"""
        try:
            self.model = joblib.load('models/kmeans_model.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            print("Models loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def predict_quality(self, milk_data):
        """Predict quality cluster for new milk samples"""
        if self.model is None or self.scaler is None:
            print("Please load models first!")
            return None
        
        # Ensure data is in correct format
        if isinstance(milk_data, dict):
            milk_data = pd.DataFrame([milk_data])
        
        # Select and scale features
        X = milk_data[self.features]
        X_scaled = self.scaler.transform(X)
        
        # Predict clusters
        clusters = self.model.predict(X_scaled)
        
        # Map clusters to quality labels
        quality_labels = {
            0: "Premium Quality",
            1: "Standard Quality", 
            2: "Average Quality",
            3: "Poor Quality",
            4: "Acidic - Needs Attention"
        }
        
        predictions = []
        for cluster in clusters:
            label = quality_labels.get(cluster, "Unknown")
            predictions.append({
                'cluster': cluster,
                'quality_label': label,
                'recommendation': self.get_recommendation(cluster, label)
            })
        
        return predictions
    
    def get_recommendation(self, cluster, label):
        """Get recommendations based on quality cluster"""
        recommendations = {
            "Premium Quality": "✅ Excellent quality! Ready for premium product lines.",
            "Standard Quality": "✅ Good quality. Suitable for standard products.",
            "Average Quality": "⚠️ Average quality. Monitor closely.",
            "Poor Quality": "❌ Poor quality. Consider rejection or reprocessing.",
            "Acidic - Needs Attention": "🚨 Acidic sample! Immediate attention required."
        }
        return recommendations.get(label, "No specific recommendation available.")
    
    def generate_test_samples(self, n_samples=50):
        """Generate test samples for validation"""
        np.random.seed(123)
        
        test_data = {
            'pH': np.random.normal(6.6, 0.3, n_samples),
            'temperature': np.random.normal(5, 3, n_samples),
            'taste': np.random.randint(1, 11, n_samples),
            'odor': np.random.randint(1, 11, n_samples),
            'fat': np.random.normal(3.5, 0.7, n_samples),
            'turbidity': np.random.normal(2.5, 1.2, n_samples),
            'color': np.random.normal(85, 15, n_samples)
        }
        
        return pd.DataFrame(test_data)
    
    def run_comprehensive_test(self):
        """Run comprehensive testing on generated samples"""
        if not self.load_models():
            return
        
        print("Generating test samples...")
        test_samples = self.generate_test_samples(100)
        
        print("Making predictions...")
        predictions = self.predict_quality(test_samples)
        
        # Add predictions to test samples
        results_df = test_samples.copy()
        results_df['predicted_cluster'] = [p['cluster'] for p in predictions]
        results_df['quality_label'] = [p['quality_label'] for p in predictions]
        results_df['recommendation'] = [p['recommendation'] for p in predictions]
        
        # Analyze results
        print("\n" + "="*50)
        print("TEST RESULTS SUMMARY")
        print("="*50)
        
        quality_distribution = results_df['quality_label'].value_counts()
        print("\nQuality Distribution:")
        for quality, count in quality_distribution.items():
            percentage = (count / len(results_df)) * 100
            print(f"  {quality}: {count} samples ({percentage:.1f}%)")
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        quality_distribution.plot(kind='bar', color='skyblue')
        plt.title('Quality Distribution in Test Samples')
        plt.xticks(rotation=45)
        plt.ylabel('Number of Samples')
        
        plt.subplot(2, 2, 2)
        plt.scatter(results_df['pH'], results_df['taste'], 
                   c=results_df['predicted_cluster'], cmap='viridis', alpha=0.7)
        plt.xlabel('pH')
        plt.ylabel('Taste Score')
        plt.title('pH vs Taste by Quality Cluster')
        plt.colorbar(label='Cluster')
        
        plt.subplot(2, 2, 3)
        results_df['fat'].hist(bins=20, alpha=0.7, color='green')
        plt.xlabel('Fat Content (%)')
        plt.ylabel('Frequency')
        plt.title('Fat Content Distribution')
        
        plt.subplot(2, 2, 4)
        # Show feature importance (using cluster centers)
        cluster_centers = self.model.cluster_centers_
        feature_importance = np.std(cluster_centers, axis=0)
        
        plt.barh(self.features, feature_importance, color='orange')
        plt.xlabel('Feature Importance (Std Dev in Cluster Centers)')
        plt.title('Feature Importance for Clustering')
        
        plt.tight_layout()
        plt.savefig('models/test_results_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nTest results visualization saved!")
        
        return results_df

if __name__ == "__main__":
    tester = MilkQualityTester()
    results = tester.run_comprehensive_test()
    
    if results is not None:
        print("\nFirst 10 test results:")
        print(results[['pH', 'taste', 'odor', 'quality_label']].head(10))