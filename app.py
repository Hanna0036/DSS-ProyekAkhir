import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from mlxtend.frequent_patterns import apriori, association_rules
import warnings

# Matikan warning agar tampilan bersih 
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

# Custom CSS untuk mempercantik
st.markdown("""
<style>
    .metric-card {background-color: #f0f2f6; border-radius: 10px; padding: 15px; margin: 10px 0;}
    .stExpander {border: 1px solid #e0e0e0; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

# =============================================
# 1. FUNGSI LOAD & FEATURE ENGINEERING 
# =============================================
@st.cache_data
def load_and_process_data(file):
    """
    Memuat data dan melakukan Feature Engineering persis seperti Notebook.
    """
    # 1. Load Data
    try:
        df = pd.read_csv(file, encoding='ISO-8859-1')
    except:
        df = pd.read_excel(file)
    
    # Bersihkan data kosong
    df = df.dropna(subset=['CustomerID', 'Description'])
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    
    # Hitung TotalPrice per baris
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    
    # 2. Tandai Retur (Invoice diawali 'C')
    df['IsReturn'] = df['InvoiceNo'].str.upper().str.startswith('C')
    
    # 3. Agregasi per Produk (StockCode)
    # Kita butuh: Total_Sales, Frequency, Revenue, Return_Rate
    df_products = df.groupby(['StockCode', 'Description']).agg(
        Total_Sales=('Quantity', lambda x: x[x > 0].sum()), # Hanya quantity positif
        Frequency=('InvoiceNo', 'nunique'),                 # Jumlah invoice unik
        Revenue=('TotalPrice', 'sum'),                      # Total pendapatan
        Total_Transactions=('InvoiceNo', 'count'),          # Total baris record
        Return_Count=('IsReturn', 'sum')                    # Jumlah kejadian retur
    ).reset_index()
    
    # 4. Hitung Rasio Retur
    df_products['Return_Rate'] = df_products['Return_Count'] / df_products['Total_Transactions']
    
    # 5. Filter: Hanya produk yang pernah terjual (Sales > 0)
    df_products = df_products[df_products['Total_Sales'] > 0]
    
    # Filter tambahan untuk kestabilan statistik (Opsional, agar tidak ada produk terjual 1x)
    # Di notebook kita pakai semua > 0, tapi untuk aplikasi sebaiknya minimal 3-5 kali terjual agar valid
    df_products = df_products[df_products['Frequency'] >= 3]
    
    return df, df_products

# =============================================
# 2. FUNGSI CLUSTERING (Sesuai Notebook: K-Means + Auto Label)
# =============================================
@st.cache_data
def perform_clustering(df, n_clusters=3):
    """
    K-Means dengan MinMaxScaler dan Auto-Labeling (Low/Average/High).
    """
    # Copy data agar aman
    df_clus = df.copy()
    
    # Ambil fitur numerik
    features = df_clus[['Total_Sales', 'Frequency', 'Revenue', 'Return_Rate']]
    
    # Normalisasi (MinMax sesuai notebook)
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Jalankan K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_clus['Cluster_ID'] = kmeans.fit_predict(features_scaled)
    
    # --- AUTO-LABELING ---
    # Hitung rata-rata Revenue per Cluster untuk tahu mana yang "Sultan"
    cluster_stats = df_clus.groupby('Cluster_ID')['Revenue'].mean().sort_values()
    
    label_mapping = {
        cluster_stats.index[0]: 'Low Performing',
        cluster_stats.index[1]: 'Average Performing',
        cluster_stats.index[2]: 'High Performing'
    }
    
    df_clus['Cluster_Label'] = df_clus['Cluster_ID'].map(label_mapping)
    
    return df_clus

# =============================================
# 3. FUNGSI TOPSIS (Sesuai Notebook)
# =============================================
def calculate_topsis(df, weights_dict):
    """
    Menghitung skor TOPSIS menggunakan bobot (dari Slider).
    """
    final_df = df.copy()
    
    # Kolom kriteria yang dipakai
    criteria = ['Total_Sales', 'Frequency', 'Revenue', 'Return_Rate']
    data_mtx = final_df[criteria]
    
    # 1. Normalisasi Vektor (x / sqrt(sum(x^2)))
    # Tambah epsilon kecil agar tidak divide by zero jika data kosong
    norm_mtx = data_mtx / np.sqrt((data_mtx**2).sum() + 1e-9)
    
    # 2. Kalikan dengan Bobot (dari Slider user)
    for col in criteria:
        norm_mtx[col] = norm_mtx[col] * weights_dict[col]
        
    # 3. Tentukan Solusi Ideal
    # Benefit: Sales, Freq, Revenue (Max is better)
    # Cost: ReturnRate (Min is better)
    ideal_pos = {
        'Total_Sales': norm_mtx['Total_Sales'].max(),
        'Frequency': norm_mtx['Frequency'].max(),
        'Revenue': norm_mtx['Revenue'].max(),
        'Return_Rate': norm_mtx['Return_Rate'].min() # Cost
    }
    
    ideal_neg = {
        'Total_Sales': norm_mtx['Total_Sales'].min(),
        'Frequency': norm_mtx['Frequency'].min(),
        'Revenue': norm_mtx['Revenue'].min(),
        'Return_Rate': norm_mtx['Return_Rate'].max()
    }
    
    # 4. Hitung Jarak & Skor
    dist_pos = np.sqrt(((norm_mtx - pd.Series(ideal_pos))**2).sum(axis=1))
    dist_neg = np.sqrt(((norm_mtx - pd.Series(ideal_neg))**2).sum(axis=1))
    
    final_df['TOPSIS_Score'] = dist_neg / (dist_pos + dist_neg + 1e-9)
    final_df['Rank'] = final_df['TOPSIS_Score'].rank(ascending=False)
    
    return final_df.sort_values('Rank')

# =============================================
# 4. FUNGSI ASSOCIATION RULE (Apriori)
# =============================================
@st.cache_data
def get_bundling_rules(df_raw, min_support=0.01, min_lift=1.2):
    """
    Market Basket Analysis untuk bundling.
    Filter 'United Kingdom' agar performa mirip notebook dan tidak out of memory.
    """
    # 1. Filter Data
    basket_data = df_raw[df_raw['Country'] == 'United Kingdom']
    
    # 2. Pivot Data
    basket = (basket_data.groupby(['InvoiceNo', 'Description'])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('InvoiceNo'))
    
    # 3. Encoding (0 atau 1)
    def encode_units(x):
        return 1 if x >= 1 else 0
    basket_sets = basket.applymap(encode_units)
    
    # Hapus POSTAGE
    if 'POSTAGE' in basket_sets.columns:
        basket_sets.drop('POSTAGE', inplace=True, axis=1)
    
    # 4. Apriori
    frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)
    
    if frequent_itemsets.empty:
        return pd.DataFrame()
        
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
    
    # Clean output (frozenset -> string)
    rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x)[0])
    rules["consequents"] = rules["consequents"].apply(lambda x: list(x)[0])
    
    return rules.sort_values('lift', ascending=False)

# =============================================
# MAIN APP
# =============================================
def main():
    st.title("🛒 Sistem DSS Pemilihan Produk Promosi E-Commerce")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("⚙️ Pengaturan")
    st.sidebar.markdown("### Upload Dataset")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload file CSV (Online Retail)",
        type=['csv', 'xlsx'],
        help="Dataset harus berisi kolom: InvoiceNo, StockCode, Description, Quantity, UnitPrice, CustomerID"
    )
    
    if uploaded_file is not None:
        # --- 1. LOAD & PROCESS ---
        with st.spinner("Memuat dan memproses data (Feature Engineering)..."):
            df_raw, df_products = load_and_process_data(uploaded_file)
        
        st.sidebar.success(f"✅ Data Valid: {len(df_products)} Produk")
        
        # --- 2. CLUSTERING ---
        with st.spinner("Menjalankan K-Means Clustering..."):
            df_clustered = perform_clustering(df_products, n_clusters=3)
            
        # --- SIDEBAR BOBOT AHP (SLIDER VERSION) ---
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ⚖️ Bobot Preferensi (AHP)")
        st.sidebar.info("Geser slider untuk menentukan prioritas kriteria.")
        
        w_revenue = st.sidebar.slider("Revenue (Uang Masuk)", 0, 10, 5, help="Seberapa penting total pendapatan?")
        w_sales = st.sidebar.slider("Total Sales (Volume)", 0, 10, 3, help="Seberapa penting jumlah barang terjual?")
        w_freq = st.sidebar.slider("Frequency (Keseringan)", 0, 10, 2, help="Seberapa penting produk sering dibeli?")
        w_return = st.sidebar.slider("Return Rate (Retur)", 0, 10, 1, help="Penalti untuk produk yang sering diretur (Makin besar = Makin benci retur)")
        
        # Hitung Total Bobot Slider
        total_w = w_revenue + w_sales + w_freq + w_return
        if total_w == 0: total_w = 1 # Hindari pembagian nol
        
        # Normalisasi Bobot agar jumlahnya 1.0 (Syarat TOPSIS)
        weights = {
            'Revenue': w_revenue / total_w,
            'Total_Sales': w_sales / total_w,
            'Frequency': w_freq / total_w,
            'Return_Rate': w_return / total_w
        }
        
        # Tampilkan bobot hasil normalisasi di sidebar
        st.sidebar.markdown("#### Bobot Ternormalisasi:")
        st.sidebar.caption(f"Revenue: {weights['Revenue']:.1%}")
        st.sidebar.caption(f"Sales: {weights['Total_Sales']:.1%}")
        st.sidebar.caption(f"Freq: {weights['Frequency']:.1%}")
        st.sidebar.caption(f"Retur: {weights['Return_Rate']:.1%}")

        # --- 3. TOPSIS ---
        with st.spinner("Menghitung Ranking TOPSIS..."):
            df_final = calculate_topsis(df_clustered, weights)
            
        # --- 4. BUNDLING (Background) ---
        with st.spinner("Mencari pola bundling (Apriori)..."):
            # Batasi data untuk demo agar cepat respons di Streamlit jika data sangat besar
            if len(df_raw) > 50000:
                rules_df = get_bundling_rules(df_raw.head(20000), min_support=0.015)
            else:
                rules_df = get_bundling_rules(df_raw, min_support=0.015)

        # =============================================
        # TABS VISUALISASI
        # =============================================
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Dashboard", 
            "🏆 Ranking TOPSIS", 
            "🔗 Association Rules", 
            "📦 Clustering", 
            "📈 Analisis Detail"
        ])
        
        # --- TAB 1: DASHBOARD ---
        with tab1:
            st.header("📊 Dashboard Overview")
            
            # Metrics Row
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Produk", len(df_final))
            col2.metric("Total Revenue", f"${df_final['Revenue'].sum():,.0f}")
            col3.metric("Rata-rata Order", f"{df_final['Frequency'].mean():.0f}")
            col4.metric("Rata-rata Retur", f"{df_final['Return_Rate'].mean()*100:.2f}%")
            
            st.markdown("---")
            
            # Scatter Plot 4 Kuadran (Plotly)
            st.subheader("Peta Posisi Produk (Sales vs Revenue)")
            
            color_map = {
                'High Performing': '#2ecc71', 
                'Average Performing': '#f1c40f', 
                'Low Performing': '#e74c3c'
            }
            
            fig = px.scatter(
                df_final, 
                x='Total_Sales', 
                y='Revenue',
                color='Cluster_Label', 
                color_discrete_map=color_map,
                hover_name='Description', 
                size='Frequency',
                log_x=True, log_y=True, 
                title="Segmentasi Produk: Sales vs Revenue (Ukuran = Frequency)",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top 10 Chart
            st.subheader("Top 10 Produk (Revenue)")
            top_rev = df_final.nlargest(10, 'Revenue')
            fig_bar = px.bar(top_rev, x='Revenue', y='Description', orientation='h', color='Cluster_Label', color_discrete_map=color_map)
            fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_bar, use_container_width=True)

        # --- TAB 2: TOPSIS RANKING (DENGAN BUNDLING) ---
        with tab2:
            st.header("🏆 Rekomendasi Produk & Bundling")
            st.info("Produk terbaik berdasarkan skor TOPSIS (Multikriteria), dilengkapi rekomendasi bundling.")
            
            # Slider jumlah produk
            top_n = st.slider("Tampilkan Top N Produk:", 5, 50, 10)
            top_products = df_final.head(top_n)
            
            for i, row in top_products.iterrows():
                # Expander untuk setiap produk
                with st.expander(f"#{int(row['Rank'])} - {row['Description']} (Skor: {row['TOPSIS_Score']:.4f})", expanded=(i==0)):
                    # Metrik Utama
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Revenue", f"${row['Revenue']:,.0f}")
                    c2.metric("Total Sales", f"{row['Total_Sales']:,.0f}")
                    c3.metric("Kategori", row['Cluster_Label'])
                    c4.metric("Return Rate", f"{row['Return_Rate']*100:.2f}%")
                    
                    # Fitur Bundling
                    st.markdown("**🎁 Rekomendasi Bundling:**")
                    
                    if not rules_df.empty:
                        # Cari aturan dimana produk ini adalah antecedent
                        my_rules = rules_df[rules_df['antecedents'] == row['Description']]
                        
                        if not my_rules.empty:
                            best = my_rules.iloc[0]
                            st.success(f"🔥 **Best Bundle:** Jual bersama **{best['consequents']}**")
                            st.caption(f"Lift: {best['lift']:.2f}x (Hubungan sangat kuat), Confidence: {best['confidence']*100:.1f}%")
                            
                            # Tampilkan opsi lain jika ada
                            if len(my_rules) > 1:
                                st.markdown("Opsi lainnya:")
                                for _, rule_row in my_rules.iloc[1:4].iterrows():
                                    st.text(f"- {rule_row['consequents']} (Lift: {rule_row['lift']:.2f})")
                        else:
                            st.info("ℹ️ Tidak ditemukan pola bundling kuat. Produk ini cenderung dibeli secara individual.")
                    else:
                        st.warning("Data association rules belum tersedia.")
            
            # Download Button
            csv = df_final.to_csv(index=False)
            st.download_button("📥 Download Laporan Lengkap (CSV)", csv, "hasil_dss_topsis.csv", "text/csv")

        # --- TAB 3: ASSOCIATION RULES (FULL TABLE) ---
        with tab3:
            st.header("🔗 Semua Pola Belanja (Apriori)")
            st.caption("Daftar lengkap aturan asosiasi yang ditemukan dari data transaksi.")
            
            if not rules_df.empty:
                st.dataframe(
                    rules_df[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(100), 
                    use_container_width=True
                )
            else:
                st.warning("Tidak ditemukan aturan asosiasi yang memenuhi syarat support/lift.")

        # --- TAB 4: CLUSTERING DETAIL ---
        with tab4:
            st.header("📦 Detail Cluster")
            
            c1, c2 = st.columns([1, 2])
            with c1:
                fig_pie = px.pie(df_final, names='Cluster_Label', title='Proporsi Jumlah Produk', 
                                color='Cluster_Label', color_discrete_map=color_map)
                st.plotly_chart(fig_pie, use_container_width=True)
            with c2:
                st.write("#### Statistik Rata-rata per Cluster")
                summary = df_final.groupby('Cluster_Label')[['Total_Sales', 'Frequency', 'Revenue', 'Return_Rate']].mean().reset_index()
                st.dataframe(summary, use_container_width=True)
            
            # Boxplot Distribusi
            st.write("#### Distribusi Revenue per Kategori")
            fig_box = px.box(df_final, x='Cluster_Label', y='Revenue', color='Cluster_Label', 
                             color_discrete_map=color_map, log_y=True)
            st.plotly_chart(fig_box, use_container_width=True)

        # --- TAB 5: ANALISIS DETAIL ---
        with tab5:
            st.header("📈 Cek Produk Spesifik")
            
            selected_prod = st.selectbox("Pilih Produk:", df_final['Description'].unique())
            prod_data = df_final[df_final['Description'] == selected_prod].iloc[0]
            
            # Tampilkan Data Raw
            st.json({
                "Description": prod_data['Description'],
                "Cluster": prod_data['Cluster_Label'],
                "TOPSIS Rank": int(prod_data['Rank']),
                "Total Sales": prod_data['Total_Sales'],
                "Revenue": prod_data['Revenue'],
                "Return Rate": f"{prod_data['Return_Rate']*100:.2f}%"
            })
            
            # Perbandingan dengan Rata-rata Toko
            avg_data = df_final[['Revenue', 'Frequency', 'Total_Sales']].mean()
            
            comp_df = pd.DataFrame({
                'Metric': ['Revenue', 'Frequency', 'Total Sales'],
                'Produk Ini': [prod_data['Revenue'], prod_data['Frequency'], prod_data['Total_Sales']],
                'Rata-rata Toko': [avg_data['Revenue'], avg_data['Frequency'], avg_data['Total_Sales']]
            })
            
            fig_comp = px.bar(comp_df, x='Metric', y=['Produk Ini', 'Rata-rata Toko'], barmode='group',
                              title="Perbandingan Performa vs Rata-rata")
            st.plotly_chart(fig_comp, use_container_width=True)

    else:
        # =============================================
        # LANDING PAGE (PENJELASAN AWAL DIKEMBALIKAN)
        # =============================================
        st.info("👆 Silakan upload dataset CSV di sidebar untuk memulai analisis")
        
        st.markdown("""
        ## 📋 Tentang Sistem Ini
        
        Sistem DSS ini dirancang untuk membantu **manajer e-commerce** dalam menentukan produk mana yang paling layak dipromosikan 
        berdasarkan analisis multi-kriteria menggunakan teknik:
        
        ### 🎯 Teknik Data Mining
        - **Association Rule Mining (Apriori)**: Menemukan pola produk yang sering dibeli bersamaan
        - **K-Means Clustering**: Mengelompokkan produk berdasarkan performa penjualan
        
        ### 🎯 Metode Decision Support System (DSS)
        - **AHP (Analytic Hierarchy Process)**: Menentukan bobot kepentingan kriteria (via Slider Preferensi)
        - **TOPSIS**: Memberikan ranking produk berdasarkan kedekatan dengan solusi ideal
        
        ### 📊 Kriteria Penilaian
        1. **Total Sales/Revenue**: Total pendapatan dari produk
        2. **Order Frequency**: Seberapa sering produk dipesan
        3. **Total Sales (Qty)**: Jumlah barang yang terjual
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
            2. Preferensi Weight (Slider)
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