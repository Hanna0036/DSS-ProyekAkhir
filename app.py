import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')

# =============================================
# KONFIGURASI HALAMAN
# =============================================
st.set_page_config(
    page_title="DSS Pemilihan Produk Promosi E-Commerce",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# FUNGSI PREPROCESSING DATA
# =============================================
@st.cache_data
def load_and_preprocess_data(file):
    """Memuat dan memproses data e-commerce"""
    
    # Load data
    df = pd.read_csv(file, encoding='ISO-8859-1')
    
    # Data cleaning
    df = df.dropna(subset=['CustomerID', 'Description'])
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df = df[~df['InvoiceNo'].str.contains('C', na=False)]  # Hapus return
    
    # Feature engineering
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    return df

@st.cache_data
def calculate_product_metrics(df):
    """Menghitung metrik untuk setiap produk"""
    
    # Agregasi per produk
    product_metrics = df.groupby(['StockCode', 'Description']).agg({
        'Quantity': 'sum',
        'TotalPrice': 'sum',
        'InvoiceNo': 'nunique',
        'CustomerID': 'nunique',
        'UnitPrice': 'mean'
    }).reset_index()
    
    product_metrics.columns = ['StockCode', 'Description', 'TotalQuantity', 
                                'TotalRevenue', 'OrderFrequency', 
                                'UniqueCustomers', 'AvgPrice']
    
    # Hitung margin (asumsi 30% dari harga)
    product_metrics['ProfitMargin'] = product_metrics['AvgPrice'] * 0.3
    product_metrics['TotalProfit'] = product_metrics['TotalQuantity'] * product_metrics['ProfitMargin']
    
    # Hitung tingkat retur (simulasi berdasarkan data historis)
    # Di sini kita simulasi return rate berdasarkan distribusi normal
    np.random.seed(42)
    product_metrics['ReturnRate'] = np.random.normal(5, 2, len(product_metrics))
    product_metrics['ReturnRate'] = product_metrics['ReturnRate'].clip(0, 15)
    
    # Ambil top 100 produk berdasarkan revenue
    product_metrics = product_metrics.nlargest(100, 'TotalRevenue')
    
    return product_metrics

# =============================================
# FUNGSI AHP (ANALYTIC HIERARCHY PROCESS)
# =============================================
def normalize_criteria(df, criteria_columns):
    """Normalisasi kriteria untuk AHP"""
    normalized_df = df[criteria_columns].copy()
    
    for col in criteria_columns:
        if col == 'ReturnRate':
            # Return rate: semakin rendah semakin baik (benefit negatif)
            normalized_df[col] = 1 / (df[col] + 1)
        else:
            # Kriteria lain: semakin tinggi semakin baik
            normalized_df[col] = df[col]
    
    # Normalisasi ke skala 0-1
    scaler = StandardScaler()
    normalized_values = scaler.fit_transform(normalized_df)
    
    # Konversi ke skala positif 0-1
    for i, col in enumerate(criteria_columns):
        min_val = normalized_values[:, i].min()
        max_val = normalized_values[:, i].max()
        if max_val - min_val > 0:
            normalized_df[col] = (normalized_values[:, i] - min_val) / (max_val - min_val)
        else:
            normalized_df[col] = 0
    
    return normalized_df

# =============================================
# FUNGSI TOPSIS
# =============================================
def calculate_topsis(df, weights):
    """Implementasi metode TOPSIS"""
    
    criteria_columns = ['TotalRevenue', 'OrderFrequency', 'TotalProfit', 'ReturnRate']
    
    # Normalisasi kriteria
    normalized = normalize_criteria(df, criteria_columns)
    
    # Weighted normalized decision matrix
    weighted_normalized = normalized.copy()
    weighted_normalized['TotalRevenue'] *= weights['TotalRevenue']
    weighted_normalized['OrderFrequency'] *= weights['OrderFrequency']
    weighted_normalized['TotalProfit'] *= weights['TotalProfit']
    weighted_normalized['ReturnRate'] *= weights['ReturnRate']
    
    # Ideal best (A+) and worst (A-) solutions
    ideal_best = weighted_normalized.max()
    ideal_worst = weighted_normalized.min()
    
    # Euclidean distance to ideal best and worst
    distance_to_best = np.sqrt(((weighted_normalized - ideal_best) ** 2).sum(axis=1))
    distance_to_worst = np.sqrt(((weighted_normalized - ideal_worst) ** 2).sum(axis=1))
    
    # TOPSIS score
    topsis_score = distance_to_worst / (distance_to_best + distance_to_worst)
    
    # Tambahkan score ke dataframe
    result_df = df.copy()
    result_df['TOPSIS_Score'] = topsis_score
    result_df = result_df.sort_values('TOPSIS_Score', ascending=False)
    result_df['Rank'] = range(1, len(result_df) + 1)
    
    return result_df

# =============================================
# FUNGSI ASSOCIATION RULE MINING (APRIORI)
# =============================================
@st.cache_data
def perform_association_analysis(df, min_support=0.01, min_confidence=0.3):
    """Melakukan analisis association rules dengan Apriori"""
    
    # Buat transaction basket
    basket = df.groupby(['InvoiceNo', 'StockCode'])['Quantity'].sum().unstack().fillna(0)
    
    # Convert ke binary (1 jika dibeli, 0 jika tidak)
    basket_binary = basket.applymap(lambda x: 1 if x > 0 else 0)
    
    # Jalankan Apriori algorithm
    frequent_itemsets = apriori(basket_binary, min_support=min_support, use_colnames=True)
    
    if len(frequent_itemsets) > 0:
        # Generate association rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        rules = rules.sort_values('lift', ascending=False)
        return rules, frequent_itemsets
    else:
        return pd.DataFrame(), pd.DataFrame()

# =============================================
# FUNGSI K-MEANS CLUSTERING
# =============================================
@st.cache_data
def perform_clustering(df, n_clusters=3):
    """Melakukan K-Means clustering pada produk"""
    
    # Pilih fitur untuk clustering
    features = df[['TotalRevenue', 'OrderFrequency', 'TotalProfit', 'ReturnRate']].copy()
    
    # Normalisasi
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(features_scaled)
    
    # Label cluster
    cluster_labels = {
        0: 'High Performer',
        1: 'Medium Performer',
        2: 'Low Performer'
    }
    
    # Urutkan cluster berdasarkan rata-rata revenue
    cluster_means = df.groupby('Cluster')['TotalRevenue'].mean().sort_values(ascending=False)
    cluster_mapping = {old: new for new, old in enumerate(cluster_means.index)}
    df['Cluster'] = df['Cluster'].map(cluster_mapping)
    df['ClusterLabel'] = df['Cluster'].map(cluster_labels)
    
    return df, kmeans

# =============================================
# MAIN APPLICATION
# =============================================
def main():
    st.title("🛒 Sistem DSS Pemilihan Produk Promosi E-Commerce")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("⚙️ Pengaturan")
    st.sidebar.markdown("### Upload Dataset")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload file CSV dari Kaggle",
        type=['csv'],
        help="Dataset harus berisi kolom: InvoiceNo, StockCode, Description, Quantity, UnitPrice, CustomerID, InvoiceDate, Country"
    )
    
    if uploaded_file is not None:
        # Load dan preprocess data
        with st.spinner("Memuat dan memproses data..."):
            df = load_and_preprocess_data(uploaded_file)
            product_metrics = calculate_product_metrics(df)
        
        st.sidebar.success(f"✅ Data berhasil dimuat: {len(df)} transaksi, {len(product_metrics)} produk")
        
        # =============================================
        # PENGATURAN BOBOT AHP
        # =============================================
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ⚖️ Bobot Kriteria AHP")
        st.sidebar.info("Total bobot harus = 100%")
        
        weight_revenue = st.sidebar.slider("Total Sales/Revenue", 0, 100, 40, 5)
        weight_frequency = st.sidebar.slider("Frekuensi Pembelian", 0, 100, 25, 5)
        weight_profit = st.sidebar.slider("Margin Keuntungan", 0, 100, 25, 5)
        weight_return = st.sidebar.slider("Tingkat Retur (Lower Better)", 0, 100, 10, 5)
        
        total_weight = weight_revenue + weight_frequency + weight_profit + weight_return
        
        if total_weight != 100:
            st.sidebar.error(f"⚠️ Total bobot = {total_weight}%. Harus 100%!")
        else:
            st.sidebar.success(f"✅ Total bobot = {total_weight}%")
        
        weights = {
            'TotalRevenue': weight_revenue / 100,
            'OrderFrequency': weight_frequency / 100,
            'TotalProfit': weight_profit / 100,
            'ReturnRate': weight_return / 100
        }
        
        # =============================================
        # TABS NAVIGATION
        # =============================================
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Dashboard",
            "🏆 Ranking TOPSIS",
            "🔗 Association Rules",
            "📦 Clustering",
            "📈 Analisis Detail"
        ])
        
        # =============================================
        # TAB 1: DASHBOARD
        # =============================================
        with tab1:
            st.header("📊 Dashboard Overview")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Produk",
                    f"{len(product_metrics)}",
                    help="Jumlah produk yang dianalisis"
                )
            
            with col2:
                total_revenue = product_metrics['TotalRevenue'].sum()
                st.metric(
                    "Total Revenue",
                    f"${total_revenue:,.0f}",
                    help="Total pendapatan dari semua produk"
                )
            
            with col3:
                avg_order_freq = product_metrics['OrderFrequency'].mean()
                st.metric(
                    "Avg Order Frequency",
                    f"{avg_order_freq:.0f}",
                    help="Rata-rata frekuensi pembelian per produk"
                )
            
            with col4:
                avg_return = product_metrics['ReturnRate'].mean()
                st.metric(
                    "Avg Return Rate",
                    f"{avg_return:.1f}%",
                    help="Rata-rata tingkat retur produk"
                )
            
            st.markdown("---")
            
            # Top 10 Products by Revenue
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top 10 Produk - Revenue")
                top_10_revenue = product_metrics.nlargest(10, 'TotalRevenue')
                
                fig = px.bar(
                    top_10_revenue,
                    x='TotalRevenue',
                    y='Description',
                    orientation='h',
                    title='Top 10 Produk Berdasarkan Revenue',
                    labels={'TotalRevenue': 'Revenue ($)', 'Description': 'Produk'}
                )
                fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Top 10 Produk - Order Frequency")
                top_10_freq = product_metrics.nlargest(10, 'OrderFrequency')
                
                fig = px.bar(
                    top_10_freq,
                    x='OrderFrequency',
                    y='Description',
                    orientation='h',
                    title='Top 10 Produk Berdasarkan Frekuensi',
                    labels={'OrderFrequency': 'Jumlah Order', 'Description': 'Produk'},
                    color='OrderFrequency',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        
        # =============================================
        # TAB 2: TOPSIS RANKING
        # =============================================
        with tab2:
            st.header("🏆 Hasil Ranking TOPSIS")
            st.info("Produk diurutkan berdasarkan skor TOPSIS yang menggabungkan semua kriteria dengan bobot yang telah ditentukan")
            
            if total_weight == 100:
                # Hitung TOPSIS
                topsis_results = calculate_topsis(product_metrics, weights)
                
                # Display top 5 recommendations
                st.subheader("✅ Top 5 Rekomendasi Produk untuk Promosi")
                
                top_5 = topsis_results.head(5)
                
                for idx, row in top_5.iterrows():
                    with st.expander(f"#{int(row['Rank'])} - {row['StockCode']}: {row['Description']}", expanded=True):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("TOPSIS Score", f"{row['TOPSIS_Score']:.4f}")
                        with col2:
                            st.metric("Revenue", f"${row['TotalRevenue']:,.0f}")
                        with col3:
                            st.metric("Order Frequency", f"{int(row['OrderFrequency'])}")
                        with col4:
                            st.metric("Return Rate", f"{row['ReturnRate']:.1f}%")
                
                st.markdown("---")
                
                # Full ranking table
                st.subheader("📋 Tabel Lengkap Ranking Produk")
                
                display_columns = ['Rank', 'StockCode', 'Description', 'TOPSIS_Score', 
                                   'TotalRevenue', 'OrderFrequency', 'TotalProfit', 'ReturnRate']
                
                display_df = topsis_results[display_columns].copy()
                display_df['TotalRevenue'] = display_df['TotalRevenue'].apply(lambda x: f"${x:,.0f}")
                display_df['TotalProfit'] = display_df['TotalProfit'].apply(lambda x: f"${x:,.0f}")
                display_df['TOPSIS_Score'] = display_df['TOPSIS_Score'].apply(lambda x: f"{x:.4f}")
                display_df['ReturnRate'] = display_df['ReturnRate'].apply(lambda x: f"{x:.1f}%")
                
                st.dataframe(display_df, use_container_width=True, height=400)
                
                # Download button
                csv = topsis_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Hasil TOPSIS (CSV)",
                    data=csv,
                    file_name="topsis_ranking.csv",
                    mime="text/csv"
                )
                
                # Visualisasi TOPSIS Score
                st.subheader("📊 Visualisasi TOPSIS Score")
                
                fig = px.bar(
                    topsis_results.head(20),
                    x='TOPSIS_Score',
                    y='Description',
                    orientation='h',
                    title='Top 20 Produk Berdasarkan TOPSIS Score',
                    labels={'TOPSIS_Score': 'TOPSIS Score', 'Description': 'Produk'},
                    color='TOPSIS_Score',
                    color_continuous_scale='blues'
                )
                fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error("⚠️ Silakan atur bobot kriteria di sidebar hingga total = 100%")
        
        # =============================================
        # TAB 3: ASSOCIATION RULES
        # =============================================
        with tab3:
            st.header("🔗 Association Rule Mining")
            st.info("Menemukan pola produk yang sering dibeli bersamaan untuk strategi bundling dan cross-selling")
            
            # Parameters
            col1, col2 = st.columns(2)
            with col1:
                min_support = st.slider("Minimum Support", 0.001, 0.05, 0.01, 0.001)
            with col2:
                min_confidence = st.slider("Minimum Confidence", 0.1, 0.9, 0.3, 0.05)
            
            with st.spinner("Menjalankan Apriori Algorithm..."):
                rules, frequent_itemsets = perform_association_analysis(df, min_support, min_confidence)
            
            if len(rules) > 0:
                st.success(f"✅ Ditemukan {len(rules)} association rules")
                
                # Top 10 Rules
                st.subheader("Top 10 Association Rules")
                
                top_rules = rules.head(10).copy()
                top_rules['antecedents'] = top_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                top_rules['consequents'] = top_rules['consequents'].apply(lambda x: ', '.join(list(x)))
                
                display_rules = top_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
                display_rules.columns = ['Produk A', 'Produk B', 'Support', 'Confidence', 'Lift']
                display_rules['Support'] = display_rules['Support'].apply(lambda x: f"{x:.4f}")
                display_rules['Confidence'] = display_rules['Confidence'].apply(lambda x: f"{x:.4f}")
                display_rules['Lift'] = display_rules['Lift'].apply(lambda x: f"{x:.2f}")
                
                st.dataframe(display_rules, use_container_width=True)
                
                # Rekomendasi Bundling
                st.subheader("💡 Rekomendasi Bundling Produk")
                
                for idx, row in top_rules.head(5).iterrows():
                    antecedents = ', '.join(list(rules.loc[idx, 'antecedents']))
                    consequents = ', '.join(list(rules.loc[idx, 'consequents']))
                    
                    st.markdown(f"""
                    **Bundle #{idx+1}:**  
                    🛍️ **Produk A:** {antecedents}  
                    🛍️ **Produk B:** {consequents}  
                    📊 **Confidence:** {row['confidence']:.2%} | **Lift:** {row['lift']:.2f}  
                    💡 *Ketika pelanggan membeli {antecedents}, ada {row['confidence']:.1%} kemungkinan mereka juga membeli {consequents}*
                    """)
                    st.markdown("---")
                  
            else:
                st.warning("⚠️ Tidak ditemukan association rules dengan parameter yang ditentukan. Coba turunkan minimum support/confidence.")
        
        # =============================================
        # TAB 4: CLUSTERING
        # =============================================
        with tab4:
            st.header("📦 K-Means Clustering")
            st.info("Mengelompokkan produk berdasarkan performa untuk strategi promosi yang berbeda")
            
            n_clusters = 3
            
            with st.spinner("Melakukan clustering..."):
                clustered_df, kmeans_model = perform_clustering(product_metrics, n_clusters)
            
            st.success(f"✅ Produk berhasil dikelompokkan ke dalam {n_clusters} cluster")
            
            # Cluster distribution
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Distribusi Cluster")
                cluster_counts = clustered_df['ClusterLabel'].value_counts()
                
                fig = px.pie(
                    values=cluster_counts.values,
                    names=cluster_counts.index,
                    title='Distribusi Produk per Cluster'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Karakteristik Cluster")
                cluster_summary = clustered_df.groupby('ClusterLabel').agg({
                    'TotalRevenue': 'mean',
                    'OrderFrequency': 'mean',
                    'TotalProfit': 'mean',
                    'ReturnRate': 'mean'
                }).round(2)
                
                cluster_summary.columns = ['Avg Revenue', 'Avg Frequency', 'Avg Profit', 'Avg Return %']
                st.dataframe(cluster_summary, use_container_width=True)
            
            # Cluster details
            st.subheader("Detail Produk per Cluster")
            
            for cluster_id in sorted(clustered_df['Cluster'].unique()):
                cluster_data = clustered_df[clustered_df['Cluster'] == cluster_id]
                cluster_label = cluster_data['ClusterLabel'].iloc[0]
                
                with st.expander(f"📦 {cluster_label} ({len(cluster_data)} produk)"):
                    st.dataframe(
                        cluster_data[['StockCode', 'Description', 'TotalRevenue', 
                                      'OrderFrequency', 'ReturnRate']].head(10),
                        use_container_width=True
                    )
                    
                    if cluster_id == 0:
                        st.success("💡 Strategi: Prioritas utama untuk promosi agresif dan maintain performa")
                    elif cluster_id == 1:
                        st.info("💡 Strategi: Tingkatkan visibilitas dengan iklan targeted dan bundle deals")
                    else:
                        st.warning("💡 Strategi: Evaluasi produk, pertimbangkan diskon atau phasing out")
        
        # =============================================
        # TAB 5: ANALISIS DETAIL
        # =============================================
        with tab5:
            st.header("📈 Analisis Detail Produk")
            
            # Product selector
            selected_product = st.selectbox(
                "Pilih Produk untuk Analisis Detail",
                product_metrics['Description'].tolist()
            )
            
            product_data = product_metrics[product_metrics['Description'] == selected_product].iloc[0]
            
            st.subheader(f"📦 {product_data['Description']}")
            st.caption(f"Stock Code: {product_data['StockCode']}")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Revenue", f"${product_data['TotalRevenue']:,.0f}")
            with col2:
                st.metric("Order Frequency", f"{int(product_data['OrderFrequency'])}")
            with col3:
                st.metric("Total Profit", f"${product_data['TotalProfit']:,.0f}")
            with col4:
                st.metric("Return Rate", f"{product_data['ReturnRate']:.1f}%")
            
            # Performance comparison
            st.subheader("Perbandingan dengan Produk Lain")
            
            metrics_comparison = pd.DataFrame({
                'Metric': ['Revenue', 'Frequency', 'Profit', 'Return Rate'],
                'Product Value': [
                    product_data['TotalRevenue'],
                    product_data['OrderFrequency'],
                    product_data['TotalProfit'],
                    product_data['ReturnRate']
                ],
                'Average Value': [
                    product_metrics['TotalRevenue'].mean(),
                    product_metrics['OrderFrequency'].mean(),
                    product_metrics['TotalProfit'].mean(),
                    product_metrics['ReturnRate'].mean()
                ]
            })
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Produk Ini',
                x=metrics_comparison['Metric'],
                y=metrics_comparison['Product Value']
            ))
            fig.add_trace(go.Bar(
                name='Rata-rata',
                x=metrics_comparison['Metric'],
                y=metrics_comparison['Average Value']
            ))
            
            fig.update_layout(
                title='Perbandingan Performa Produk',
                barmode='group',
                yaxis_title='Nilai'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Sales trend (jika ada data temporal)
            st.subheader("Trend Penjualan")
            product_sales = df[df['Description'] == selected_product].copy()
            product_sales['YearMonth'] = product_sales['InvoiceDate'].dt.to_period('M').astype(str)
            
            monthly_sales = product_sales.groupby('YearMonth').agg({
                'Quantity': 'sum',
                'TotalPrice': 'sum'
            }).reset_index()
            
            fig = px.line(
                monthly_sales,
                x='YearMonth',
                y='TotalPrice',
                title=f'Trend Penjualan Bulanan - {selected_product}',
                labels={'YearMonth': 'Bulan', 'TotalPrice': 'Total Penjualan ($)'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Landing page
        st.info("👆 Silakan upload dataset CSV di sidebar untuk memulai analisis")
        
        st.markdown("""
        ## 📋 Tentang Sistem Ini
        
        Sistem DSS ini dirancang untuk membantu **manajer e-commerce** dalam menentukan produk mana yang paling layak dipromosikan 
        berdasarkan analisis multi-kriteria menggunakan teknik:
        
        ### 🎯 Teknik Data Mining
        - **Association Rule Mining (Apriori)**: Menemukan pola produk yang sering dibeli bersamaan
        - **K-Means Clustering**: Mengelompokkan produk berdasarkan performa penjualan
        
        ### 🎯 Metode Decision Support System (DSS)
        - **AHP (Analytic Hierarchy Process)**: Menentukan bobot kepentingan kriteria
        - **TOPSIS**: Memberikan ranking produk berdasarkan kedekatan dengan solusi ideal
        
        ### 📊 Kriteria Penilaian
        1. **Total Sales/Revenue**: Total pendapatan dari produk
        2. **Order Frequency**: Seberapa sering produk dipesan
        3. **Profit Margin**: Keuntungan yang dihasilkan
        4. **Return Rate**: Tingkat pengembalian produk (semakin rendah semakin baik)
        
        ### 📁 Format Dataset
        Dataset harus dalam format CSV dengan kolom:
        - `InvoiceNo`: Nomor invoice
        - `StockCode`: Kode produk
        - `Description`: Deskripsi produk
        - `Quantity`: Jumlah produk yang dibeli
        - `UnitPrice`: Harga per unit
        - `CustomerID`: ID pelanggan
        - `InvoiceDate`: Tanggal transaksi
        - `Country`: Negara pelanggan
        
        ### 🚀 Cara Menggunakan
        1. Download dataset dari [Kaggle E-Commerce Data](https://www.kaggle.com/datasets/carrie1/ecommerce-data)
        2. Upload file CSV melalui sidebar
        3. Atur bobot kriteria sesuai prioritas bisnis Anda
        4. Jelajahi hasil analisis melalui berbagai tab
        
        ---
        
        💡 **Tip**: Sistem ini akan otomatis memproses data dan memberikan rekomendasi produk yang paling layak untuk dipromosikan!
        """)
        
        # Tampilkan preview metodologi
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🔍 Data Mining Workflow
            ```
            1. Data Preprocessing
               ↓
            2. Feature Engineering
               ↓
            3. Association Rule Mining
               ↓
            4. K-Means Clustering
               ↓
            5. Insight Generation
            ```
            """)
        
        with col2:
            st.markdown("""
            ### 🎯 DSS Workflow
            ```
            1. Define Criteria
               ↓
            2. AHP Weight Assignment
               ↓
            3. Data Normalization
               ↓
            4. TOPSIS Calculation
               ↓
            5. Product Ranking
            ```
            """)

if __name__ == "__main__":
    main()