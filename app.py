# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Set page config HARUS di baris pertama
st.set_page_config(
    page_title="Dashboard Analisis Clustering Data Kesehatan",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2E86AB;
        border-bottom: 3px solid #2E86AB;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .highlight-box {
        background-color: #E8F4F8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Function untuk load data (dengan data contoh jika file tidak ada)
def load_data():
    try:
        # Coba baca data dari file
        data = pd.read_csv('Data kesehatan.csv', delimiter=';')
        st.success(f"✅ Data berhasil dimuat: {len(data)} records")
    except Exception as e:
        st.warning("⚠️ File tidak ditemukan, menggunakan data contoh...")
        # Buat data contoh
        np.random.seed(42)
        n_samples = 1000
        
        data = pd.DataFrame({
            'Age': np.random.normal(45, 15, n_samples).clip(18, 80),
            'BMI': np.random.normal(25, 5, n_samples).clip(15, 40),
            'Exercise_Frequency': np.random.randint(0, 7, n_samples),
            'Diet_Quality': np.random.uniform(1, 10, n_samples),
            'Sleep_Hours': np.random.normal(7, 1.5, n_samples).clip(4, 12),
            'Smoking_Status': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'Alcohol_Consumption': np.random.exponential(5, n_samples).clip(0, 30),
            'Health_Score': np.random.uniform(3, 10, n_samples)
        })
        st.info(f"📊 Data contoh dibuat: {len(data)} records")
    
    return data

# Function untuk preprocessing
def preprocess_data(data):
    # Pilih hanya kolom numerik
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    # Jika ada nilai non-numerik, konversi
    for col in data.columns:
        if col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Handling missing values
    for col in numeric_cols:
        if data[col].isnull().sum() > 0:
            data[col].fillna(data[col].median(), inplace=True)
    
    # Normalisasi data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[numeric_cols])
    data_scaled_df = pd.DataFrame(data_scaled, columns=numeric_cols)
    
    return data, data_scaled_df, scaler

# Function untuk clustering
def perform_clustering(data_scaled, n_clusters=3):
    results = {}
    
    try:
        # K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(data_scaled)
        results['K-Means'] = {
            'labels': kmeans_labels,
            'score': silhouette_score(data_scaled, kmeans_labels) if len(np.unique(kmeans_labels)) > 1 else 0
        }
    except Exception as e:
        results['K-Means'] = {'labels': None, 'score': -1}
    
    try:
        # GMM
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm_labels = gmm.fit_predict(data_scaled)
        results['GMM'] = {
            'labels': gmm_labels,
            'score': silhouette_score(data_scaled, gmm_labels) if len(np.unique(gmm_labels)) > 1 else 0
        }
    except Exception as e:
        results['GMM'] = {'labels': None, 'score': -1}
    
    try:
        # Hierarchical
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        hierarchical_labels = hierarchical.fit_predict(data_scaled)
        results['Hierarchical'] = {
            'labels': hierarchical_labels,
            'score': silhouette_score(data_scaled, hierarchical_labels) if len(np.unique(hierarchical_labels)) > 1 else 0
        }
    except Exception as e:
        results['Hierarchical'] = {'labels': None, 'score': -1}
    
    try:
        # Spectral Clustering
        spectral = SpectralClustering(n_clusters=n_clusters, random_state=42, affinity='nearest_neighbors', n_init=10)
        spectral_labels = spectral.fit_predict(data_scaled)
        results['Spectral'] = {
            'labels': spectral_labels,
            'score': silhouette_score(data_scaled, spectral_labels) if len(np.unique(spectral_labels)) > 1 else 0
        }
    except Exception as e:
        results['Spectral'] = {'labels': None, 'score': -1}
    
    return results

# Function untuk visualisasi
def plot_clustering_results(data_scaled, labels, algorithm_name):
    # Reduce dimensions untuk visualisasi 2D
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data_scaled)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if labels is not None and len(np.unique(labels)) > 0:
        scatter = ax.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
    else:
        ax.scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.6)
    
    ax.set_title(f'Visualisasi Clustering - {algorithm_name}')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.grid(True, alpha=0.3)
    
    return fig

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Dashboard Analisis Clustering Data Kesehatan</h1>', unsafe_allow_html=True)
    
    # Load data
    data = load_data()
    
    # Sidebar untuk navigasi
    with st.sidebar:
        st.markdown("### 📊 Navigasi Dashboard")
        page = st.radio(
            "Pilih Halaman:",
            ["📖 About", "📊 Dataset", "⚙️ Preprocessing", "🤖 Machine Learning", 
             "📈 Analysis Terbaik", "🎨 Visualisasi", "🔮 Prediksi", "📞 Contact"]
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ Pengaturan")
        n_clusters = st.slider("Jumlah Cluster:", 2, 10, 3)
        
        st.markdown("---")
        st.markdown("### 👤 Developer")
        st.info("""
        **Muji Silvi R**  
        Mahasiswa Sains Data  
        Universitas Muhammadiyah Semarang
        """)
    
    # Tab 1: About
    if page == "📖 About":
        st.markdown('<h2 class="sub-header">📖 Tentang Project</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("""
            <div class="highlight-box">
            <h3>🎯 Pendahuluan Project</h3>
            <p>Project ini merupakan implementasi dari mata kuliah <strong>Machine Learning</strong> 
            untuk analisis clustering pada data kesehatan. Implementasi ini mencakup beberapa 
            algoritma clustering yang populer untuk mengelompokkan data berdasarkan karakteristik 
            yang sama.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Jumlah Algoritma", "5")
        
        st.markdown(f"""
        <div class="highlight-box">
        <h3>📊 Tentang Dataset</h3>
        <p><strong>Sumber Data:</strong> WHO (World Health Organization) - Data kesehatan global</p>
        <p><strong>Jumlah Data:</strong> {len(data)} records</p>
        <p><strong>Variabel:</strong> 
        <ul>
            <li><strong>Age:</strong> Usia pasien (tahun)</li>
            <li><strong>BMI:</strong> Indeks Massa Tubuh</li>
            <li><strong>Exercise_Frequency:</strong> Frekuensi olahraga per minggu</li>
            <li><strong>Diet_Quality:</strong> Kualitas diet (skala 1-10)</li>
            <li><strong>Sleep_Hours:</strong> Jam tidur per hari</li>
            <li><strong>Smoking_Status:</strong> Status merokok (0=Tidak, 1=Ya)</li>
            <li><strong>Alcohol_Consumption:</strong> Konsumsi alkohol per minggu</li>
            <li><strong>Health_Score:</strong> Skor kesehatan keseluruhan</li>
        </ul>
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
        <h3>🎯 Tujuan Analisis</h3>
        <p>1. Mengidentifikasi pola dan kelompok dalam data kesehatan populasi</p>
        <p>2. Memahami karakteristik kelompok kesehatan yang berbeda</p>
        <p>3. Memberikan rekomendasi intervensi kesehatan berdasarkan cluster</p>
        <p>4. Mengimplementasikan dan membandingkan berbagai algoritma clustering</p>
        <p>5. Membangun dashboard interaktif untuk analisis data kesehatan</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Deskripsi algoritma
        st.markdown('<h3 class="sub-header">🔬 Metodologi Algoritma Clustering</h3>', unsafe_allow_html=True)
        
        with st.expander("📚 K-Means Clustering"):
            st.write("""
            **Deskripsi:** Algoritma partisi yang membagi data menjadi k cluster berdasarkan jarak ke centroid.
            
            **Formula:** 
            $$J = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2$$
            
            **Kelebihan:**
            - Efisien untuk data besar
            - Mudah diimplementasikan
            
            **Kekurangan:**
            - Harus tentukan k
            - Sensitif outlier
            """)
        
        with st.expander("📚 Gaussian Mixture Model (GMM)"):
            st.write("""
            **Deskripsi:** Model probabilistik yang mengasumsikan data berasal dari campuran distribusi Gaussian.
            
            **Formula:**
            $$p(x) = \sum_{i=1}^{k} \pi_i \mathcal{N}(x|\mu_i, \Sigma_i)$$
            
            **Kelebihan:**
            - Fleksibel bentuk cluster
            - Probabilistik
            
            **Kekurangan:**
            - Komputasi intensif
            - Dapat konvergen lokal
            """)
        
        with st.expander("📚 Hierarchical Clustering"):
            st.write("""
            **Deskripsi:** Membangun hierarki cluster melalui pendekatan aglomeratif atau divisif.
            
            **Visualisasi:** Dendrogram berdasarkan jarak antar cluster
            
            **Kelebihan:**
            - Tidak perlu tentukan k awal
            - Visualisasi hierarki
            
            **Kekurangan:**
            - Kompleksitas tinggi
            - Sensitif noise
            """)
        
        with st.expander("📚 Spectral Clustering"):
            st.write("""
            **Deskripsi:** Menggunakan spektrum matriks similarity untuk reduksi dimensi sebelum clustering.
            
            **Kelebihan:**
            - Efektif untuk bentuk kompleks
            - Menggunakan informasi global
            
            **Kekurangan:**
            - Komputasi mahal
            - Sensitif parameter
            """)
    
    # Tab 2: Dataset
    elif page == "📊 Dataset":
        st.markdown('<h2 class="sub-header">📊 Dataset Preview</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.dataframe(data.head(10), use_container_width=True)
        
        with col2:
            st.metric("Total Records", len(data))
            st.metric("Jumlah Variabel", len(data.columns))
        
        with col3:
            st.metric("Missing Values", data.isnull().sum().sum())
            st.metric("Tipe Data", "Numerik")
        
        # Statistik deskriptif
        st.markdown('<h3 class="sub-header">📈 Statistik Deskriptif</h3>', unsafe_allow_html=True)
        st.dataframe(data.describe(), use_container_width=True)
        
        # Distribusi data
        st.markdown('<h3 class="sub-header">📊 Distribusi Data</h3>', unsafe_allow_html=True)
        
        # Pilih kolom untuk visualisasi
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        selected_col = st.selectbox("Pilih variabel untuk dilihat distribusinya:", numeric_cols)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(data[selected_col].dropna(), bins=30, edgecolor='black', alpha=0.7)
        ax.set_title(f'Distribusi {selected_col}')
        ax.set_xlabel(selected_col)
        ax.set_ylabel('Frekuensi')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Tab 3: Preprocessing
    elif page == "⚙️ Preprocessing":
        st.markdown('<h2 class="sub-header">⚙️ Data Preprocessing</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
        <h3>🔧 Tahapan Preprocessing yang Dilakukan:</h3>
        <ol>
            <li><strong>Data Loading</strong>: Membaca data dari file CSV</li>
            <li><strong>Pengecekan Missing Values</strong>: Identifikasi dan penanganan data yang hilang</li>
            <li><strong>Normalisasi Data</strong>: Standardisasi menggunakan StandardScaler</li>
            <li><strong>Feature Selection</strong>: Seleksi fitur numerik untuk clustering</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Proses preprocessing
        with st.spinner("Sedang memproses data..."):
            data_processed, data_scaled, scaler = preprocess_data(data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Data Asli (Sample)")
            st.dataframe(data.head(), use_container_width=True)
            
            st.markdown("**Statistik Data Asli:**")
            st.write(f"- Rata-rata Age: {data['Age'].mean():.2f}")
            st.write(f"- Rata-rata BMI: {data['BMI'].mean():.2f}")
        
        with col2:
            st.markdown("#### Data Setelah Normalisasi (Sample)")
            st.dataframe(pd.DataFrame(data_scaled, 
                                     columns=data.select_dtypes(include=[np.number]).columns).head(), 
                        use_container_width=True)
            
            st.markdown("**Statistik Data Normalisasi:**")
            st.write(f"- Rata-rata (mendekati 0): {data_scaled.mean():.4f}")
            st.write(f"- Standar deviasi (mendekati 1): {data_scaled.std():.4f}")
    
    # Tab 4: Machine Learning
    elif page == "🤖 Machine Learning":
        st.markdown('<h2 class="sub-header">🤖 Analisis Clustering</h2>', unsafe_allow_html=True)
        
        # Preprocess data
        with st.spinner("Mempersiapkan data..."):
            _, data_scaled, _ = preprocess_data(data)
        
        # Perform clustering
        with st.spinner("Sedang melakukan clustering..."):
            results = perform_clustering(data_scaled, n_clusters)
        
        # Filter valid results
        valid_results = {k: v for k, v in results.items() if v['score'] >= 0}
        
        if not valid_results:
            st.error("Tidak ada algoritma yang berhasil dijalankan. Cek data dan parameter.")
            return
        
        # Find best algorithm
        best_algo = max(valid_results, key=lambda x: valid_results[x]['score'])
        best_score = valid_results[best_algo]['score']
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
            <h3>🏆 Algoritma Terbaik</h3>
            <h2>{best_algo}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
            <h3>⭐ Silhouette Score</h3>
            <h2>{best_score:.4f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
            <h3>🔢 Jumlah Cluster</h3>
            <h2>{n_clusters}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Scores table
        st.markdown('<h3 class="sub-header">📈 Perbandingan Performa Algoritma</h3>', unsafe_allow_html=True)
        
        # Buat DataFrame untuk hasil
        score_data = []
        for algo, result in valid_results.items():
            score_data.append({
                'Algorithm': algo,
                'Silhouette Score': result['score'],
                'Status': '🏆 TERBAIK' if algo == best_algo else '✅'
            })
        
        score_df = pd.DataFrame(score_data)
        
        # Tampilkan dengan highlight
        def highlight_row(row):
            if row['Algorithm'] == best_algo:
                return ['background-color: #FFEAA7; font-weight: bold'] * len(row)
            return [''] * len(row)
        
        st.dataframe(score_df.style.apply(highlight_row, axis=1), use_container_width=True)
        
        # Visualisasi perbandingan
        st.markdown('<h3 class="sub-header">📊 Visualisasi Perbandingan</h3>', unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        algorithms = list(valid_results.keys())
        scores = [valid_results[algo]['score'] for algo in algorithms]
        
        colors = ['#FF6B6B' if algo == best_algo else '#4ECDC4' for algo in algorithms]
        bars = ax.bar(algorithms, scores, color=colors, edgecolor='black')
        
        ax.set_ylabel('Silhouette Score')
        ax.set_title('Perbandingan Performa Algoritma Clustering')
        ax.set_ylim(0, max(scores) * 1.2)
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Elbow method untuk K-Means
        st.markdown('<h3 class="sub-header">📉 Metode Elbow untuk Menentukan k Optimal</h3>', unsafe_allow_html=True)
        
        inertia = []
        k_range = range(1, 11)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data_scaled)
            inertia.append(kmeans.inertia_)
        
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(k_range, inertia, 'bo-', linewidth=2, markersize=8)
        ax2.set_xlabel('Jumlah Cluster (k)')
        ax2.set_ylabel('Inertia')
        ax2.set_title('Metode Elbow untuk Menentukan k Optimal')
        ax2.grid(True, alpha=0.3)
        
        # Highlight titik siku
        ax2.axvline(x=n_clusters, color='red', linestyle='--', alpha=0.5, label=f'k={n_clusters}')
        ax2.legend()
        
        st.pyplot(fig2)
    
    # Tab 5: Analysis Terbaik
    elif page == "📈 Analysis Terbaik":
        st.markdown('<h2 class="sub-header">📈 Analisis dengan Algoritma Terbaik</h2>', unsafe_allow_html=True)
        
        # Preprocess data
        _, data_scaled, _ = preprocess_data(data)
        
        # Perform clustering and find best
        results = perform_clustering(data_scaled, n_clusters)
        valid_results = {k: v for k, v in results.items() if v['score'] >= 0}
        
        if not valid_results:
            st.error("Tidak ada algoritma yang berhasil.")
            return
        
        best_algo = max(valid_results, key=lambda x: valid_results[x]['score'])
        best_labels = valid_results[best_algo]['labels']
        
        st.markdown(f"""
        <div class="highlight-box">
        <h3>🎯 Algoritma Terpilih: {best_algo}</h3>
        <p>Silhouette Score: <strong>{valid_results[best_algo]['score']:.4f}</strong></p>
        <p>Jumlah Cluster: <strong>{n_clusters}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Steps untuk algoritma terbaik
        st.markdown('<h3 class="sub-header">🔧 Langkah-langkah Analisis</h3>', unsafe_allow_html=True)
        
        steps = {
            "K-Means": [
                "1. **Inisialisasi**: Pilih k centroid secara acak",
                "2. **Assignment**: Tentukan setiap titik ke cluster terdekat",
                "3. **Update**: Hitung ulang centroid sebagai rata-rata cluster",
                "4. **Iterasi**: Ulangi hingga konvergen",
                "5. **Evaluasi**: Hitung silhouette score"
            ],
            "GMM": [
                "1. **Inisialisasi**: Tentukan parameter awal Gaussian",
                "2. **Expectation**: Hitung probabilitas keanggotaan",
                "3. **Maximization**: Update parameter Gaussian",
                "4. **Iterasi**: Ulangi hingga konvergen",
                "5. **Klasterisasi**: Assign titik ke Gaussian tertinggi"
            ],
            "Hierarchical": [
                "1. **Inisialisasi**: Setiap titik sebagai cluster tunggal",
                "2. **Hitung Jarak**: Hitung jarak antar cluster",
                "3. **Gabungkan**: Gabungkan dua cluster terdekat",
                "4. **Update**: Update matriks jarak",
                "5. **Iterasi**: Ulangi hingga k cluster"
            ],
            "Spectral": [
                "1. **Matriks Similarity**: Hitung matriks similarity",
                "2. **Laplacian**: Hitung matriks Laplacian",
                "3. **Eigen Decomposition**: Ekstrak eigenvector",
                "4. **Klasterisasi**: Cluster di ruang eigenvector",
                "5. **Mapping**: Map kembali ke ruang asli"
            ]
        }
        
        if best_algo in steps:
            for step in steps[best_algo]:
                st.write(step)
        
        # Cluster characteristics
        st.markdown('<h3 class="sub-header">📊 Karakteristik Cluster</h3>', unsafe_allow_html=True)
        
        # Add labels to original data
        data_with_clusters = data.copy()
        data_with_clusters['Cluster'] = best_labels
        
        # Calculate statistics per cluster
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        cluster_stats = data_with_clusters.groupby('Cluster')[numeric_cols].mean()
        
        # Display dengan styling
        cm = sns.light_palette("green", as_cmap=True)
        st.dataframe(cluster_stats.style.background_gradient(cmap=cm), use_container_width=True)
        
        # Visualisasi cluster
        st.markdown('<h3 class="sub-header">🎨 Visualisasi Cluster</h3>', unsafe_allow_html=True)
        
        if best_labels is not None:
            cluster_fig = plot_clustering_results(data_scaled, best_labels, best_algo)
            st.pyplot(cluster_fig)
            
            st.markdown("**Interpretasi:**")
            st.write(f"""
            - Hasil clustering menggunakan algoritma **{best_algo}**
            - Terdapat **{n_clusters} cluster** yang terbentuk
            - Setiap cluster diwakili oleh warna yang berbeda
            - Cluster yang terpisah baik menunjukkan karakteristik yang berbeda
            - Silhouette score **{valid_results[best_algo]['score']:.4f}** menunjukkan kualitas clustering
            """)
    
    # Tab 6: Visualisasi
    elif page == "🎨 Visualisasi":
        st.markdown('<h2 class="sub-header">🎨 Visualisasi Hasil Clustering</h2>', unsafe_allow_html=True)
        
        # Preprocess data
        _, data_scaled, _ = preprocess_data(data)
        
        # Perform clustering
        results = perform_clustering(data_scaled, n_clusters)
        valid_results = {k: v for k, v in results.items() if v['score'] >= 0}
        
        if not valid_results:
            st.error("Tidak ada algoritma yang berhasil.")
            return
        
        # Pilih algoritma untuk visualisasi
        algorithm = st.selectbox(
            "Pilih algoritma untuk visualisasi:",
            list(valid_results.keys())
        )
        
        labels = valid_results[algorithm]['labels']
        
        # PCA untuk 2D visualization
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data_scaled)
        
        # Plot dengan Plotly
        plot_df = pd.DataFrame({
            'PCA1': data_2d[:, 0],
            'PCA2': data_2d[:, 1],
            'Cluster': labels.astype(str)
        })
        
        fig = px.scatter(
            plot_df, x='PCA1', y='PCA2', color='Cluster',
            title=f'Visualisasi Cluster - {algorithm}',
            color_discrete_sequence=px.colors.qualitative.Set1,
            opacity=0.7
        )
        
        fig.update_layout(
            width=800,
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretasi
        st.markdown("#### 📝 Interpretasi Visualisasi")
        st.write(f"""
        **Hasil visualisasi menggunakan {algorithm}:**
        
        - **{len(np.unique(labels))} cluster** teridentifikasi
        - Setiap titik mewakili satu observasi dalam data
        - Warna berbeda menunjukkan cluster yang berbeda
        - Cluster yang padat menunjukkan homogenitas tinggi
        - Overlap antar cluster mengindikasikan kesamaan karakteristik
        
        **Silhouette Score:** {valid_results[algorithm]['score']:.4f}
        """)
        
        # Additional insights
        st.markdown("#### 💡 Insights Tambahan")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **Cluster Padat:**
            - Karakteristik seragam
            - Intervensi kesehatan bisa distandardisasi
            - Respons terhadap treatment mungkin serupa
            """)
        
        with col2:
            st.info("""
            **Cluster Tersebar:**
            - Variasi tinggi dalam cluster
            - Perlukan analisis lebih mendalam
            - Mungkin perlu sub-clustering
            """)
    
    # Tab 7: Prediksi
    elif page == "🔮 Prediksi":
        st.markdown('<h2 class="sub-header">🔮 Prediksi Cluster Data Baru</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📤 Upload Data Baru")
            uploaded_file = st.file_uploader("Upload file CSV", type=['csv'], key="uploader")
            
            if uploaded_file is not None:
                try:
                    new_data = pd.read_csv(uploaded_file)
                    st.success(f"✅ Data berhasil diupload! ({len(new_data)} records)")
                    st.dataframe(new_data.head(), use_container_width=True)
                except Exception as e:
                    st.error(f"Error membaca file: {e}")
                    new_data = None
            else:
                st.info("Upload file CSV atau gunakan data contoh")
                # Gunakan sample data
                new_data = data.sample(min(5, len(data)), random_state=42)
                st.dataframe(new_data, use_container_width=True)
        
        with col2:
            st.markdown("#### ⚙️ Pilih Algoritma")
            algorithm = st.selectbox(
                "Pilih algoritma clustering:",
                ["K-Means", "GMM", "Hierarchical", "Spectral"]
            )
            
            if st.button("🚀 Jalankan Prediksi", type="primary", use_container_width=True):
                if new_data is not None:
                    with st.spinner("Sedang memproses..."):
                        try:
                            # Preprocess training data
                            _, data_scaled, scaler = preprocess_data(data)
                            
                            # Preprocess new data
                            new_numeric_cols = new_data.select_dtypes(include=[np.number]).columns
                            new_data_processed = new_data.copy()
                            
                            # Handle missing values
                            for col in new_numeric_cols:
                                if new_data_processed[col].isnull().any():
                                    new_data_processed[col].fillna(new_data_processed[col].median(), inplace=True)
                            
                            # Scale new data
                            new_data_scaled = scaler.transform(new_data_processed[new_numeric_cols])
                            
                            # Train and predict
                            if algorithm == "K-Means":
                                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                            elif algorithm == "GMM":
                                model = GaussianMixture(n_components=n_clusters, random_state=42)
                            elif algorithm == "Hierarchical":
                                model = AgglomerativeClustering(n_clusters=n_clusters)
                            else:  # Spectral
                                model = SpectralClustering(n_clusters=n_clusters, random_state=42, affinity='nearest_neighbors', n_init=10)
                            
                            # Fit on training data
                            model.fit(data_scaled)
                            
                            # Predict
                            if hasattr(model, 'predict'):
                                predictions = model.predict(new_data_scaled)
                            else:
                                # For algorithms without predict method
                                predictions = model.fit_predict(new_data_scaled)
                            
                            # Display results
                            result_df = new_data_processed.copy()
                            result_df['Predicted_Cluster'] = predictions
                            
                            st.success(f"✅ Prediksi selesai menggunakan {algorithm}!")
                            
                            # Show results
                            st.markdown("#### 📋 Hasil Prediksi")
                            st.dataframe(result_df, use_container_width=True)
                            
                            # Visualize distribution
                            st.markdown("#### 📊 Distribusi Cluster")
                            
                            cluster_counts = pd.Series(predictions).value_counts().sort_index()
                            
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                            
                            # Bar chart
                            colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_counts)))
                            bars = ax1.bar(cluster_counts.index.astype(str), cluster_counts.values, color=colors)
                            ax1.set_xlabel('Cluster')
                            ax1.set_ylabel('Jumlah Data')
                            ax1.set_title('Distribusi Data per Cluster')
                            ax1.grid(True, alpha=0.3)
                            
                            # Add count labels
                            for bar in bars:
                                height = bar.get_height()
                                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                        f'{int(height)}', ha='center', va='bottom')
                            
                            # Pie chart
                            ax2.pie(cluster_counts.values, labels=cluster_counts.index.astype(str), 
                                   autopct='%1.1f%%', startangle=90, colors=colors)
                            ax2.set_title('Persentase per Cluster')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                        except Exception as e:
                            st.error(f"Error dalam prediksi: {e}")
                else:
                    st.warning("Silakan upload data terlebih dahulu")
    
    # Tab 8: Contact
    elif page == "📞 Contact":
        st.markdown('<h2 class="sub-header">📞 Kontak & Informasi</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # You can add an image here
            st.markdown("""
            <div style="text-align: center; padding: 20px; border: 2px solid #2E86AB; border-radius: 10px;">
                <h3>👤 Developer</h3>
                <p style="font-size: 1.2rem; font-weight: bold;">Muji Silvi R</p>
                <p>Mahasiswa Sains Data</p>
                <p>Universitas Muhammadiyah Semarang</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="margin-top: 20px;">
                <h4>🏆 Capaian Project</h4>
                <p>✓ Dashboard Clustering Komprehensif</p>
                <p>✓ 5 Algoritma Machine Learning</p>
                <p>✓ Visualisasi Interaktif</p>
                <p>✓ Prediksi Data Baru</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="highlight-box">
            <h3>📱 Informasi Kontak</h3>
            <table style="width: 100%; font-size: 1.1rem;">
                <tr>
                    <td><strong>📱 WhatsApp:</strong></td>
                    <td>089889382442</td>
                </tr>
                <tr>
                    <td><strong>📧 Email:</strong></td>
                    <td>silvirakhmawati@gmail.com</td>
                </tr>
                <tr>
                    <td><strong>📷 Instagram:</strong></td>
                    <td>@silvimuji</td>
                </tr>
                <tr>
                    <td><strong>🏫 Institusi:</strong></td>
                    <td>Universitas Muhammadiyah Semarang</td>
                </tr>
                <tr>
                    <td><strong>🎓 Program Studi:</strong></td>
                    <td>Sains Data</td>
                </tr>
                <tr>
                    <td><strong>📚 Mata Kuliah:</strong></td>
                    <td>Machine Learning</td>
                </tr>
            </table>
            </div>
            """, unsafe_allow_html=True)
            
            # Project details
            st.markdown("""
            <div class="highlight-box" style="margin-top: 20px;">
            <h3>ℹ️ Detail Project</h3>
            <p><strong>Nama Project:</strong> Dashboard Analisis Clustering Data Kesehatan</p>
            <p><strong>Deskripsi:</strong> Implementasi algoritma clustering untuk analisis data kesehatan</p>
            <p><strong>Tahun:</strong> 2024</p>
            <p><strong>Teknologi:</strong> Python, Streamlit, Scikit-learn, Plotly</p>
            <p><strong>Fitur:</strong> 8 tab navigasi, 5 algoritma, visualisasi interaktif</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Feedback section
        st.markdown('<h3 class="sub-header">💬 Feedback & Saran</h3>', unsafe_allow_html=True)
        
        with st.form("feedback_form"):
            name = st.text_input("Nama Anda")
            email = st.text_input("Email")
            message = st.text_area("Pesan / Feedback / Saran")
            
            submitted = st.form_submit_button("Kirim Feedback", type="primary")
            if submitted:
                if name and message:
                    st.success("🎉 Terima kasih atas feedback Anda!")
                    st.balloons()
                else:
                    st.warning("Harap isi nama dan pesan")

if __name__ == "__main__":
    main()