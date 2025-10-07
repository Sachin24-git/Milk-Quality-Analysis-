import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

class MilkQualityCluster:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.optimal_k = None
        self.features = None
        
    def load_data(self, file_path='data/milk_quality.csv'):
        """Load and prepare the milk quality data"""
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Select features for clustering
        self.features = ['pH', 'temperature', 'taste', 'odor', 'fat', 'turbidity', 'color']
        X = df[self.features]
        
        return df, X
    
    def find_optimal_clusters(self, X, max_k=10):
        """Determine optimal number of clusters using elbow method and silhouette score"""
        wcss = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, kmeans.labels_))
        
        # Find optimal k (elbow point)
        differences = np.diff(wcss)
        second_derivatives = np.diff(differences)
        self.optimal_k = np.argmax(second_derivatives) + 3  # +3 because of double diff
        
        # Ensure optimal_k is within range
        self.optimal_k = max(2, min(self.optimal_k, max_k))
        
        # Plot results
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(k_range, wcss, 'bo-')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')
        plt.title('Elbow Method')
        plt.axvline(x=self.optimal_k, color='red', linestyle='--', label=f'Optimal k={self.optimal_k}')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(k_range, silhouette_scores, 'ro-')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Analysis')
        plt.axvline(x=self.optimal_k, color='red', linestyle='--', label=f'Optimal k={self.optimal_k}')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('models/cluster_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return self.optimal_k, wcss, silhouette_scores
    
    def train_model(self, X, n_clusters=None):
        """Train the K-means model"""
        if n_clusters is None:
            n_clusters = self.optimal_k
            
        # Scale the features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train K-means model
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = self.model.fit_predict(X_scaled)
        
        # Calculate silhouette score
        score = silhouette_score(X_scaled, clusters)
        print(f"Model trained with {n_clusters} clusters")
        print(f"Silhouette Score: {score:.3f}")
        
        return clusters, score
    
    def analyze_clusters(self, df, clusters):
        """Analyze and interpret each cluster"""
        df['cluster'] = clusters
        
        cluster_analysis = df.groupby('cluster').agg({
            'pH': ['mean', 'std'],
            'temperature': ['mean', 'std'],
            'taste': ['mean', 'std'],
            'odor': ['mean', 'std'],
            'fat': ['mean', 'std'],
            'turbidity': ['mean', 'std'],
            'color': ['mean', 'std'],
            'batch_id': 'count'
        }).round(3)
        
        # Name clusters based on characteristics
        cluster_names = []
        for cluster_num in range(len(cluster_analysis)):
            cluster_data = df[df['cluster'] == cluster_num]
            
            # Determine cluster characteristics
            avg_taste = cluster_data['taste'].mean()
            avg_odor = cluster_data['odor'].mean()
            avg_pH = cluster_data['pH'].mean()
            avg_fat = cluster_data['fat'].mean()
            
            if avg_taste > 7 and avg_odor > 7 and avg_pH >= 6.5:
                name = "Premium Quality"
            elif avg_taste > 5 and avg_odor > 5:
                name = "Standard Quality"
            elif avg_pH < 6.4:
                name = "Acidic - Needs Attention"
            elif avg_taste < 4 or avg_odor < 4:
                name = "Poor Quality"
            else:
                name = "Average Quality"
                
            cluster_names.append(name)
        
        cluster_analysis['segment_name'] = cluster_names
        
        return df, cluster_analysis
    
    def save_model(self):
        """Save the trained model and scaler"""
        if not os.path.exists('models'):
            os.makedirs('models')
            
        joblib.dump(self.model, 'models/kmeans_model.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        print("Model and scaler saved successfully!")
    
    def run_training(self):
        """Complete training pipeline"""
        print("Loading data...")
        df, X = self.load_data()
        
        print("Finding optimal clusters...")
        optimal_k, wcss, silhouette_scores = self.find_optimal_clusters(X)
        
        print(f"Training model with {optimal_k} clusters...")
        clusters, score = self.train_model(X, optimal_k)
        
        print("Analyzing clusters...")
        df, cluster_analysis = self.analyze_clusters(df, clusters)
        
        print("Saving model...")
        self.save_model()
        
        # Save clustered data
        df.to_csv('data/milk_quality_clustered.csv', index=False)
        
        return df, cluster_analysis, score

if __name__ == "__main__":
    trainer = MilkQualityCluster()
    df, analysis, score = trainer.run_training()
    
    print("\nTraining completed!")
    print(f"Final Silhouette Score: {score:.3f}")
    print("\nCluster Analysis:")
    print(analysis)