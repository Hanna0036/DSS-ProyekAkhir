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
    .stExpander {border: 1px solid #e0e0e0; border-radius: 8px; padding: 5px; box-shadow: 0 2px 4px 0 rgba(0,0,0,0.05);}
    
    /* Mengubah warna font pada metric value */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #2c3e50; /* Darker text - ini akan bekerja di Light Theme */
    }
    
    /* Mengubah warna font pada Nilai Metrik (angka besar) */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #f7931a; /* Orange/Gold terang yang kontras untuk angka */
    }
    
    /* Menambah sedikit padding di tab */
    .stTabs [data-baseweb="tab-list"] button {
        padding: 10px;
    }
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
        
        # Penggunaan Key untuk memastikan slider punya identitas unik
        w_revenue = st.sidebar.slider("Revenue (Uang Masuk) 💰", 0, 10, 5, help="Seberapa penting total pendapatan?", key='w_revenue')
        w_sales = st.sidebar.slider("Total Sales (Volume) 📦", 0, 10, 3, help="Seberapa penting jumlah barang terjual?", key='w_sales')
        w_freq = st.sidebar.slider("Frequency (Keseringan) 🔄", 0, 10, 2, help="Seberapa penting produk sering dibeli?", key='w_freq')
        w_return = st.sidebar.slider("Return Rate (Retur) ❌", 0, 10, 1, help="Penalti untuk produk yang sering diretur (Makin besar = Makin benci retur)", key='w_return')
        
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
        st.sidebar.code(f"""
Revenue:    {weights['Revenue']:.1%}
Sales:      {weights['Total_Sales']:.1%}
Frequency:  {weights['Frequency']:.1%}
Return Rate: {weights['Return_Rate']:.1%}
Total:      {(weights['Revenue'] + weights['Total_Sales'] + weights['Frequency'] + weights['Return_Rate']):.1%}
        """)

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

        if rules_df.empty:
            st.sidebar.warning("⚠️ Tidak ditemukan Rules Bundling kuat.")        

# =============================================
        # TABS VISUALISASI
        # =============================================
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Dashboard Overview", 
            "🏆 Rekomendasi TOPSIS", 
            "🔗 Association Rules", 
            "📦 Clustering Detail", 
            "📈 Analisis Detail Produk"
        ])
        
        # --- TAB 1: DASHBOARD ---
        with tab1:
            st.header("📊 Dashboard Performa Produk")
            
            # Metrics Row (dipercantik dengan kolom)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Produk Analisis", len(df_final))
            with col2:
                st.metric("Total Revenue", f"${df_final['Revenue'].sum():,.0f}")
            with col3:
                st.metric("Rata-rata Order/Produk", f"{df_final['Frequency'].mean():.0f}")
            with col4:
                st.metric("Rata-rata Retur", f"{df_final['Return_Rate'].mean()*100:.2f}%")
            
            st.markdown("---")
            
            # Scatter Plot 4 Kuadran (Plotly)
            st.subheader("Peta Posisi Produk (Sales vs Revenue)")
            st.caption("Ukuran lingkaran menunjukkan Frekuensi (Keseringan Order)")
            
            color_map = {
                'High Performing': '#2ecc71', # Hijau
                'Average Performing': '#f1c40f', # Kuning
                'Low Performing': '#e74c3c' # Merah
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
                title="Segmentasi Produk: Total Sales vs Revenue",
                height=550
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top 10 Chart
            st.subheader("Top 10 Produk (Revenue Terbesar)")
            top_rev = df_final.nlargest(10, 'Revenue')
            fig_bar = px.bar(
                top_rev, 
                x='Revenue', 
                y='Description', 
                orientation='h', 
                color='Cluster_Label', 
                color_discrete_map=color_map,
                title="10 Produk dengan Revenue Tertinggi"
            )
            fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_bar, use_container_width=True)

        # --- TAB 2: TOPSIS RANKING (DENGAN BUNDLING) ---
        with tab2:
            st.header("🏆 Rekomendasi Produk Promosi TOPSIS")
            st.markdown("Berikut adalah produk-produk yang direkomendasikan untuk promosi berdasarkan bobot preferensi Anda. (Skor TOPSIS = Semakin Tinggi Semakin Baik).")
            
            # Slider jumlah produk
            top_n = st.slider("Tampilkan Top N Produk Rekomendasi:", 5, 50, 10, key='top_n_topsis')
            top_products = df_final.head(top_n)
            
            for i, row in top_products.iterrows():
                # Expander untuk setiap produk
                # Penggunaan f-string dan Markdown untuk judul yang lebih menarik
                cluster_emoji = "💎" if row['Cluster_Label'] == 'High Performing' else "🌟" if row['Cluster_Label'] == 'Average Performing' else "📉"
                with st.expander(f"**{cluster_emoji} #{int(row['Rank'])}** | **{row['Description']}** (Cluster: **{row['Cluster_Label']}** | Skor TOPSIS: **{row['TOPSIS_Score']:.4f}**)", expanded=(i==0)):
                    
                    # Metrik Utama dengan kolom
                    st.markdown("#### Detail Performa Kunci:")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Revenue", f"${row['Revenue']:,.0f}")
                    c2.metric("Total Sales (Qty)", f"{row['Total_Sales']:,.0f}")
                    c3.metric("Order Frequency", f"{row['Frequency']:,.0f}")
                    c4.metric("Return Rate", f"{row['Return_Rate']*100:.2f}%", delta_color="inverse")
                    
                    st.markdown("---")
                    
                    # Fitur Bundling
                    st.markdown("#### 🎁 Rekomendasi Bundling (Market Basket Analysis):")
                    
                    if not rules_df.empty:
                        # Cari aturan dimana produk ini adalah antecedent
                        my_rules = rules_df[rules_df['antecedents'] == row['Description']]
                        
                        if not my_rules.empty:
                            best = my_rules.iloc[0]
                            st.success(f"**🔥 Best Bundle:** Pelanggan yang beli **{row['Description']}** cenderung beli **{best['consequents']}**")
                            
                            # Tampilkan detail metrik Apriori
                            st.markdown(f"""
                            * **Lift:** `{best['lift']:.2f}x` (Hubungan sangat kuat. > 1.0 = saling menguatkan)
                            * **Confidence:** `{best['confidence']*100:.1f}%` (Jika A dibeli, seberapa sering B ikut dibeli)
                            * **Support:** `{best['support']*100:.2f}%` (Seberapa sering A dan B muncul bersamaan)
                            """)
                            
                            # Tampilkan opsi lain jika ada
                            if len(my_rules) > 1:
                                st.markdown("Opsi Bundling Lainnya (Top 3):")
                                for _, rule_row in my_rules.iloc[1:4].iterrows():
                                    st.text(f"- {rule_row['consequents']} (Lift: {rule_row['lift']:.2f}, Confidence: {rule_row['confidence']*100:.1f}%)")
                        else:
                            st.info("ℹ️ Tidak ditemukan pola bundling kuat. Produk ini cenderung dibeli secara individual atau rules-nya di bawah threshold.")
                    else:
                        st.warning("Data association rules belum tersedia atau tidak memenuhi syarat support/lift.")
            
            st.markdown("---")
            # Download Button
            csv = df_final.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Laporan Lengkap (CSV)", 
                csv, 
                "hasil_dss_topsis.csv", 
                "text/csv"
            )

        # --- TAB 3: ASSOCIATION RULES (FULL TABLE) ---
        with tab3:
            st.header("🔗 Semua Pola Belanja (Apriori)")
            st.caption("Daftar lengkap aturan asosiasi yang ditemukan. Filter berdasarkan **Lift** (kekuatan hubungan) dan **Confidence** (probabilitas A -> B).")
            
            if not rules_df.empty:
                st.dataframe(
                    rules_df[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'leverage']].head(100), 
                    use_container_width=True,
                    column_config={
                        "support": st.column_config.NumberColumn("Support (%)", format="%.2f"),
                        "confidence": st.column_config.NumberColumn("Confidence (%)", format="%.2f"),
                        "lift": st.column_config.NumberColumn("Lift (x)", format="%.2f"),
                        "leverage": st.column_config.NumberColumn("Leverage", format="%.4f")
                    }
                )
            else:
                st.warning("Tidak ditemukan aturan asosiasi yang memenuhi syarat support/lift yang ditentukan.")

        # --- TAB 4: CLUSTERING DETAIL ---
        with tab4:
            st.header("📦 Detail K-Means Clustering")
            st.caption("Pengelompokan produk berdasarkan 4 kriteria: Sales, Frequency, Revenue, dan Return Rate.")
            
            col_pie, col_sum = st.columns([1, 2])
            with col_pie:
                fig_pie = px.pie(
                    df_final, 
                    names='Cluster_Label', 
                    title='Proporsi Jumlah Produk per Cluster', 
                    color='Cluster_Label', 
                    color_discrete_map=color_map
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            with col_sum:
                st.write("#### Statistik Rata-rata Fitur per Cluster")
                summary = df_final.groupby('Cluster_Label')[['Total_Sales', 'Frequency', 'Revenue', 'Return_Rate']].mean().reset_index()
                # Sort agar High Performing di atas
                order_map = {'High Performing': 3, 'Average Performing': 2, 'Low Performing': 1}
                summary['Order'] = summary['Cluster_Label'].map(order_map)
                summary = summary.sort_values('Order', ascending=False).drop('Order', axis=1)
                
                st.dataframe(
                    summary.set_index('Cluster_Label').style.format({
                        'Revenue': "Rp {:,.0f}",
                        'Return_Rate': "{:.2%}"
                    }), 
                    use_container_width=True
                )
            
            # Boxplot Distribusi
            st.markdown("---")
            st.write("#### Distribusi Revenue per Kategori (Skala Logaritmik)")
            fig_box = px.box(
                df_final, 
                x='Cluster_Label', 
                y='Revenue', 
                color='Cluster_Label', 
                color_discrete_map=color_map, 
                log_y=True,
                title="Perbandingan Distribusi Revenue Antar Cluster"
            )
            st.plotly_chart(fig_box, use_container_width=True)

        # --- TAB 5: ANALISIS DETAIL (Fokus Peningkatan Tampilan) ---
        with tab5:
            st.header("📈 Analisis Detail Produk Spesifik")
            st.caption("Lihat metrik kunci dan perbandingan kinerja produk individual dengan rata-rata toko.")
            
            # Selectbox di kolom yang lebih lebar
            col_select, _ = st.columns([2, 1])
            with col_select:
                selected_prod = st.selectbox("Pilih Produk yang Ingin Dicek:", df_final['Description'].unique(), key='prod_select')
            
            # Pastikan produk terpilih ada datanya
            if selected_prod:
                prod_data = df_final[df_final['Description'] == selected_prod].iloc[0]
                avg_data = df_final[['Revenue', 'Frequency', 'Total_Sales', 'Return_Rate']].mean()
                
                st.markdown("---")
                
                # Seksion 1: Ringkasan Kunci (Card Metric)
                st.subheader("Ringkasan Kinerja Produk")
                
                c1, c2, c3, c4, c5 = st.columns(5)
                
                with c1:
                    st.metric("TOPSIS Rank", f"#{int(prod_data['Rank'])}")
                with c2:
                    st.metric("Cluster Kategori", prod_data['Cluster_Label'])
                with c3:
                    st.metric("Total Revenue", f"${prod_data['Revenue']:,.0f}")
                with c4:
                    st.metric("Total Sales (Qty)", f"{prod_data['Total_Sales']:,.0f}")
                with c5:
                    # Hitung delta return rate vs average
                    delta_return = (prod_data['Return_Rate'] - avg_data['Return_Rate']) * 100
                    st.metric(
                        "Return Rate", 
                        f"{prod_data['Return_Rate']*100:.2f}%", 
                        delta=f"{delta_return:.2f}% vs Rata-rata",
                        delta_color="inverse"
                    )

                st.markdown("---")
                
                # Seksion 2: Perbandingan Bar Chart
                st.subheader("Perbandingan Performa vs Rata-rata Toko")
                
                comp_df = pd.DataFrame({
                    'Metric': ['Revenue', 'Frequency', 'Total Sales'],
                    'Produk Ini': [prod_data['Revenue'], prod_data['Frequency'], prod_data['Total_Sales']],
                    'Rata-rata Toko': [avg_data['Revenue'], avg_data['Frequency'], avg_data['Total_Sales']]
                })
                
                fig_comp = px.bar(
                    comp_df, 
                    x='Metric', 
                    y=['Produk Ini', 'Rata-rata Toko'], 
                    barmode='group',
                    title="Komparasi Metrik Kinerja Produk",
                    color_discrete_map={'Produk Ini': '#3498db', 'Rata-rata Toko': '#bdc3c7'}
                )
                st.plotly_chart(fig_comp, use_container_width=True)

    else:
        # =============================================
        # LANDING PAGE (PENJELASAN AWAL DIKEMBALIKAN)
        # (Beberapa pemformatan dipercantik)
        # =============================================
        st.info("👆 Silakan upload dataset CSV/XLSX di sidebar untuk memulai analisis")
        
        st.markdown("""
        ## 📋 Tentang Sistem Ini
        
        Sistem DSS ini dirancang untuk membantu **manajer e-commerce** dalam menentukan produk mana yang paling layak dipromosikan 
        berdasarkan analisis multi-kriteria menggunakan teknik:
        
        ### 🎯 Teknik Data Mining
        * **Association Rule Mining (Apriori)**: Menemukan pola produk yang sering dibeli bersamaan.
        * **K-Means Clustering**: Mengelompokkan produk berdasarkan performa penjualan (Revenue, Sales, Frequency, Return Rate).
        
        ### 🎯 Metode Decision Support System (DSS)
        * **AHP (Analytic Hierarchy Process) Sederhana**: Menentukan bobot kepentingan kriteria (via Slider Preferensi).
        * **TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)**: Memberikan ranking produk berdasarkan kedekatan dengan solusi ideal.
        
        ### 📊 Kriteria Penilaian
        | Kriteria | Tipe | Deskripsi |
        | :--- | :--- | :--- |
        | **Revenue** | Benefit | Total pendapatan dari produk (Max is better) |
        | **Total Sales (Qty)** | Benefit | Jumlah barang yang terjual (Max is better) |
        | **Order Frequency** | Benefit | Seberapa sering produk dipesan (Max is better) |
        | **Return Rate** | Cost | Tingkat pengembalian produk (Min is better) |
        
        ### 📁 Format Dataset
        Dataset harus dalam format CSV/XLSX dengan kolom kunci: `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `UnitPrice`, `CustomerID`.
        
        ### 🚀 Cara Menggunakan
        1. Upload file CSV/XLSX melalui sidebar.
        2. Atur **Bobot Preferensi** (Revenue, Sales, dll.) sesuai prioritas bisnis Anda di sidebar.
        3. Jelajahi hasil analisis di tab **"Rekomendasi TOPSIS"** untuk produk terbaik dan saran bundling.
        
        ---
        
        💡 **Tip**: Gunakan tab **"Analisis Detail Produk"** untuk menginspeksi produk secara spesifik.
        """)
        
        # Tampilkan preview metodologi
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🔍 Data Mining Workflow
            ```mermaid
            graph TD
                A[Data Load & Clean] --> B(Feature Engineering: RFM-like);
                B --> C(K-Means Clustering);
                A --> D(Apriori: Market Basket);
                C & D --> E(Insight Generation);
            ```
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            ### 🎯 DSS Workflow
            ```mermaid
            graph TD
                A[Product Features] --> B{Bobot Preferensi (Slider)};
                B --> C[TOPSIS Calculation];
                C --> D(Final Product Ranking);
            ```
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()