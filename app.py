import graphviz
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
    page_title="Smart E-Commerce Decision System",
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
        color: #2c3e50;
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
    
    # Filter tambahan untuk kestabilan statistik
    df_products = df_products[df_products['Frequency'] >= 3]
    
    return df, df_products

# =============================================
# 2. FUNGSI CLUSTERING (K-Means + Auto Label)
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
# 3. FUNGSI TOPSIS
# =============================================
def calculate_topsis(df, weights_dict):
    """
    Menghitung skor TOPSIS menggunakan bobot langsung (dari Slider).
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
    st.title("🛒 Smart E-Commerce Decision System")
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
            
        # --- SIDEBAR BOBOT (DIRECT WEIGHTING) ---
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ⚖️ Konfigurasi Bobot Kriteria")
        st.sidebar.info("Geser slider untuk menentukan prioritas kriteria secara manual.")
        
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
Revenue:     {weights['Revenue']:.1%}
Sales:       {weights['Total_Sales']:.1%}
Frequency:   {weights['Frequency']:.1%}
Return Rate: {weights['Return_Rate']:.1%}
Total:       {(weights['Revenue'] + weights['Total_Sales'] + weights['Frequency'] + weights['Return_Rate']):.1%}
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
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Dashboard Overview", 
            "🏆 Rekomendasi TOPSIS", 
            "🔗 Ide Paket Bundling", 
            "📦 Kategori Produk", 
            "📈 Analisis Detail Produk",
            "💡 Rekomendasi Strategis"
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
                with st.expander(f"**{int(row['Rank'])}** | **{row['Description']}** (Cluster: **{row['Cluster_Label']}** | Skor TOPSIS: **{row['TOPSIS_Score']:.4f}**)", expanded=(i==0)):
                    
                    # Metrik Utama dengan kolom
                    st.markdown("#### Detail Performa Kunci:")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Revenue", f"${row['Revenue']:,.0f}")
                    c2.metric("Total Sales (Qty)", f"{row['Total_Sales']:,.0f}")
                    c3.metric("Order Frequency", f"{row['Frequency']:,.0f}")
                    c4.metric("Return Rate", f"{row['Return_Rate']*100:.2f}%", delta_color="inverse")
                    
                    st.markdown("---")
                    
                    # Fitur Bundling
                    st.markdown("####  Rekomendasi Bundling (Market Basket Analysis):")
                    
                    if not rules_df.empty:
                        # Cari aturan dimana produk ini adalah antecedent
                        my_rules = rules_df[rules_df['antecedents'] == row['Description']]
                        
                        if not my_rules.empty:
                            best = my_rules.iloc[0]
                            st.success(f"**Best Bundle**: Pelanggan yang beli {row['Description']} cenderung beli {best['consequents']}")
                            
                            # Tampilkan detail metrik Apriori4
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
            st.header("🔗 Ide Paket & Bundling")
            st.markdown("""
            Halaman ini menunjukkan **Pasangan Produk** yang sering dibeli bersamaan oleh pelanggan. 
            Gunakan data ini untuk membuat strategi:
            * **Paket Hemat** (Diskon jika beli keduanya).
            * **Cross-Selling** (Tawarkan produk B saat pelanggan melihat produk A).
            
            """)
            if not rules_df.empty:
                    # Membuat dataframe yang lebih mudah dibaca orang awam
                    display_rules = rules_df[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
                    display_rules.columns = ['Jika Beli Barang A', 'Maka Beli Barang B', 'Frekuensi Muncul (%)', 'Peluang Beli (%)', 'Kekuatan Hubungan (x)']
                    
                    # Format persentase
                    display_rules['Frekuensi Muncul (%)'] = (display_rules['Frekuensi Muncul (%)'] * 100).map('{:.2f}%'.format)
                    display_rules['Peluang Beli (%)'] = (display_rules['Peluang Beli (%)'] * 100).map('{:.1f}%'.format)
                    display_rules['Kekuatan Hubungan (x)'] = display_rules['Kekuatan Hubungan (x)'].map('{:.2f}x'.format)
                    
                    st.dataframe(
                        display_rules.head(100), 
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    st.info("""
                    **Cara Membaca Tabel:**
                    * **Peluang Beli:** Jika orang beli A, berapa persen kemungkinan dia beli B?
                    * **Kekuatan Hubungan:** Angka di atas 1.0 berarti mereka memang saling berhubungan, bukan kebetulan.
                    """)
            else:
                    st.warning("Tidak ditemukan pola yang kuat pada data ini. Coba gunakan data dengan rentang waktu lebih lama.")
        # --- TAB 4: CLUSTERING DETAIL ---
        with tab4:
            st.header("📦 Pengelompokan Produk (Segmentasi)")
            st.caption("Pengelompokan produk dengan K-Means Clsutering berdasarkan 4 kriteria: Sales, Frequency, Revenue, dan Return Rate.")
            
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

        # --- TAB 5: DETAIL PRODUK (GABUNGAN METRIK + RADAR + BAR) ---
        with tab5:
            st.header("📈 Analisis Detail & Komparasi Produk")
            st.caption("Analisis mendalam satu produk dibandingkan dengan rata-rata cluster dan rata-rata toko.")
            
            # --- 1. FILTER CLUSTER & PILIH PRODUK ---
            col_filter, col_select = st.columns([1, 2])
            
            with col_filter:
                # Tambahkan opsi "Semua Cluster"
                cluster_options = ["Semua Cluster"] + sorted(df_final['Cluster_Label'].unique().tolist())
                selected_cluster_filter = st.selectbox("Filter Kategori Cluster:", cluster_options, key='cluster_filter_tab5')
            
            # Filter DataFrame berdasarkan pilihan cluster
            if selected_cluster_filter != "Semua Cluster":
                filtered_df_prod = df_final[df_final['Cluster_Label'] == selected_cluster_filter]
            else:
                filtered_df_prod = df_final
            
            with col_select:
                selected_prod = st.selectbox("Pilih Produk:", filtered_df_prod['Description'].unique(), key='prod_select_tab5')
            
            if selected_prod:
                prod_data = df_final[df_final['Description'] == selected_prod].iloc[0]
                
                # Data Pembanding (Benchmark)
                # A. Rata-rata Toko (Global)
                avg_store = df_final[['Revenue', 'Frequency', 'Total_Sales', 'Return_Rate']].mean()
                
                # B. Rata-rata Cluster (Kelompoknya sendiri)
                c_label = prod_data['Cluster_Label']
                avg_cluster = df_final[df_final['Cluster_Label'] == c_label][['Revenue', 'Frequency', 'Total_Sales', 'Return_Rate']].mean()
                
                # Max Values untuk Normalisasi Radar
                max_vals = df_final[['Revenue', 'Frequency', 'Total_Sales', 'Return_Rate']].max()
                
                st.markdown("---")
                
                # --- BAGIAN 1: KARTU METRIK UTAMA ---
                st.subheader(f"Ringkasan Kinerja: {selected_prod}")
                c1, c2, c3, c4, c5 = st.columns(5)
                
                with c1: st.metric("TOPSIS Rank", f"#{int(prod_data['Rank'])}")
                with c2: st.metric("Cluster", prod_data['Cluster_Label'])
                with c3: st.metric("Revenue", f"${prod_data['Revenue']:,.0f}")
                with c4: st.metric("Sales Qty", f"{prod_data['Total_Sales']:,.0f}")
                
                delta_ret = (prod_data['Return_Rate'] - avg_store['Return_Rate']) * 100
                with c5: 
                    st.metric("Return Rate", f"{prod_data['Return_Rate']*100:.2f}%", 
                             delta=f"{delta_ret:.2f}% vs Rata-rata Toko", delta_color="inverse")
                
                st.markdown("---")
                
                # --- BAGIAN 2: GRAFIK GABUNGAN (RADAR & BAR) ---
                col_radar, col_bar = st.columns(2)
                
                # A. RADAR CHART (Produk vs Rata-rata Cluster)
                with col_radar:
                    metrics_radar = ['Total_Sales', 'Frequency', 'Revenue', 'Return_Rate']
                    labels_radar = ['Sales Vol', 'Frequency', 'Revenue', 'Quality (Low Return)']
                    
                    # Normalisasi (0-1)
                    def get_radar_val(row_dat, mx_dat):
                        return [
                            row_dat['Total_Sales'] / (mx_dat['Total_Sales'] + 1e-9),
                            row_dat['Frequency'] / (mx_dat['Frequency'] + 1e-9),
                            row_dat['Revenue'] / (mx_dat['Revenue'] + 1e-9),
                            1 - (row_dat['Return_Rate'] / (mx_dat['Return_Rate'] + 0.001))
                        ]
                    
                    val_prod = get_radar_val(prod_data, max_vals)
                    val_clust = get_radar_val(avg_cluster, max_vals)
                    
                    # Warna Rata-rata Cluster (Benchmark) - Diset Hijau agar kontras
                    benchmark_color = '#2ecc71' 
                    
                    fig_radar = go.Figure()
                    # Produk Ini (Biru)
                    fig_radar.add_trace(go.Scatterpolar(
                        r=val_prod, theta=labels_radar, fill='toself', 
                        name='Produk Ini',
                        line_color='#3498db'
                    ))
                    # Rata-rata Cluster (Warna sesuai Cluster)
                    fig_radar.add_trace(go.Scatterpolar(
                        r=val_clust, theta=labels_radar, fill='toself', 
                        name=f'Rata-rata {c_label}',
                        line_color=benchmark_color
                    ))
                    
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        title=f"Radar: Produk vs Cluster {c_label}",
                        margin=dict(t=80, b=80, l=60, r=60),
                        height=400
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                    st.info(f"ℹ️ Grafik ini membandingkan produk dengan rata-rata kelompok **{c_label}** (Area Hijau).")

                # B. BAR CHART (Produk vs Rata-rata Toko) - Existing Feature
                with col_bar:
                    comp_df = pd.DataFrame({
                        'Metric': ['Revenue', 'Frequency', 'Total Sales'],
                        'Produk Ini': [prod_data['Revenue'], prod_data['Frequency'], prod_data['Total_Sales']],
                        'Rata-rata Toko': [avg_store['Revenue'], avg_store['Frequency'], avg_store['Total_Sales']]
                    })
                    
                    fig_bar = px.bar(
                        comp_df, x='Metric', y=['Produk Ini', 'Rata-rata Toko'], 
                        barmode='group', title="Bar: Produk vs Rata-rata Seluruh Toko",
                        color_discrete_map={'Produk Ini': '#3498db', 'Rata-rata Toko': '#bdc3c7'},
                        height=400
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                    st.caption("Grafik batang menunjukkan nilai absolut untuk membandingkan skala produk terhadap rata-rata umum toko.")

# --- TAB 6: REKOMENDASI STRATEGIS (FINAL) ---
        with tab6:
            st.header("💡 Rekomendasi Strategis & Simulasi")
            
            # --- BAGIAN 1: DETEKSI MASALAH & PELUANG (SMART INSIGHTS) ---
            st.subheader("1. Deteksi Masalah & Peluang")
            st.caption("Analisis otomatis untuk menemukan produk yang memerlukan perhatian khusus (Hidden Gem, High Return, Dead Stock).")
            
            # Logika deteksi spesifik
            def get_specific_insight(row):
                insights = []
                # Cek Retur Tinggi (Top 25% rate retur)
                if row['Return_Rate'] > df_final['Return_Rate'].quantile(0.75):
                    insights.append("⚠️ **High Return Alert**: Cek kualitas fisik atau deskripsi produk karena tingkat pengembalian tinggi.")
                
                # Cek Potensi Hidden Gem (Rank TOPSIS tinggi tapi Sales masih rendah/Average)
                # Misal: Masuk Top 20 ranking, tapi cluster label bukan 'High Performing'
                if row['Rank'] <= 20 and row['Cluster_Label'] != 'High Performing':
                    insights.append("💎 **Hidden Gem**: Skor performa keseluruhan sangat bagus tapi belum masuk kategori 'High Performing'. Perlu boost marketing.")
                
                # Cek Dead Stock Potential (Bottom 10% frequency)
                if row['Frequency'] < df_final['Frequency'].quantile(0.10):
                    insights.append("📉 **Dead Stock Risk**: Pergerakan sangat lambat. Pertimbangkan diskon cuci gudang.")
                
                return list(set(insights))

            # Filter produk yang punya insight khusus
            df_final['Actionable_Insights'] = df_final.apply(get_specific_insight, axis=1)
            df_insights = df_final[df_final['Actionable_Insights'].map(len) > 0].sort_values('Rank')
            
            if not df_insights.empty:
                st.info(f"Sistem mendeteksi **{len(df_insights)} produk** dengan anomali atau potensi khusus:")
                
                # Flatten list insight menjadi string untuk display di tabel
                df_insights['Insight_Text'] = df_insights['Actionable_Insights'].apply(lambda x: " | ".join(x))
                
                st.dataframe(
                    df_insights[['Description', 'Cluster_Label', 'Return_Rate', 'Insight_Text']],
                    column_config={
                        "Return_Rate": st.column_config.NumberColumn("Retur", format="%.2f%%"),
                        "Insight_Text": st.column_config.TextColumn("Rekomendasi Spesifik", width="large"),
                    },
                    use_container_width=True
                )
            else:
                st.success("✅ Semua produk terlihat normal, tidak ada anomali ekstrim yang terdeteksi.")

            st.markdown("---")

            # --- BAGIAN 2: SIMULATOR DISKON ---
            st.subheader("2. 🧮 Simulator Diskon (Promo Planner)")
            st.caption("Hitung target penjualan (Break-even Volume) jika Anda memberikan diskon, agar Revenue tidak turun.")
            
            c_sim1, c_sim2 = st.columns([1, 2])
            
            with c_sim1:
                st.markdown("#### Input Simulasi")
                target_product = st.selectbox("Pilih Produk:", df_final['Description'].unique(), key='sim_prod')
                
                # Hitung harga rata-rata saat ini (Revenue / Sales)
                # Ditambah epsilon 1e-9 agar tidak error divide by zero
                current_price = df_final[df_final['Description'] == target_product]['Revenue'].values[0] / \
                                (df_final[df_final['Description'] == target_product]['Total_Sales'].values[0] + 1e-9)
                
                discount_percent = st.slider("Rencana Diskon (%)", 0, 50, 10, step=5)
                
            with c_sim2:
                # Ambil data produk
                prod_sim_data = df_final[df_final['Description'] == target_product].iloc[0]
                curr_sales = prod_sim_data['Total_Sales']
                curr_rev = prod_sim_data['Revenue']
                
                # Kalkulasi
                new_price = current_price * (1 - (discount_percent/100))
                
                # Target Sales = Revenue Lama / Harga Baru
                target_sales_qty = curr_rev / new_price if new_price > 0 else 0
                additional_qty_needed = target_sales_qty - curr_sales
                percent_increase_needed = (additional_qty_needed / curr_sales) * 100 if curr_sales > 0 else 0
                
                # Tampilan Hasil
                st.markdown("#### Hasil Analisis Break-even")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Harga Diskon", f"${new_price:.2f}", delta=f"-{discount_percent}%", delta_color="inverse")
                m2.metric("Target Qty", f"{target_sales_qty:,.0f}", delta=f"+{additional_qty_needed:,.0f} Unit")
                m3.metric("Kenaikan Vol. Wajib", f"{percent_increase_needed:.1f}%")
                
                st.info(f"💡 Jika diskon **{discount_percent}%** diberikan, Anda wajib menaikkan penjualan sebesar **{percent_increase_needed:.1f}%** hanya untuk mendapatkan uang yang sama (Revenue ${curr_rev:,.0f}).")
                
                # Visualisasi Break-even Simple Bar
                sim_df = pd.DataFrame({
                    'Skenario': ['Saat Ini', 'Target Diskon'],
                    'Quantity': [curr_sales, target_sales_qty]
                })
                fig_sim = px.bar(sim_df, x='Skenario', y='Quantity', text_auto='.0f',
                                title="Perbandingan Volume Penjualan yang Dibutuhkan",
                                color='Skenario', color_discrete_sequence=['#95a5a6', '#e74c3c'])
                st.plotly_chart(fig_sim, use_container_width=True)

            st.markdown("---")

            # --- BAGIAN 3: STRATEGI BUNDLING (ASSOCIATION RULES) ---
            st.subheader("3. 🔗 Strategi Bundling & Cross-Selling")
            st.caption("Rekomendasi pasangan produk berdasarkan pola belanja pelanggan (Market Basket Analysis).")

            if not rules_df.empty:
                # Ambil Top 10 Rules berdasarkan Lift (Kekuatan Hubungan)
                top_rules = rules_df.sort_values('lift', ascending=False).head(10)
                
                col_rules_viz, col_rules_list = st.columns([2, 1])
                
                with col_rules_viz:
                    st.markdown("#### 🕸️ Peta Koneksi Produk")
                    # Visualisasi Graphviz
                    graph = graphviz.Digraph()
                    graph.attr(rankdir='LR', size='10')
                    graph.attr('node', shape='box', style='filled', fillcolor='#e1f5fe', color='#0277bd', fontname='Arial', fontsize='10')
                    graph.attr('edge', fontname='Arial', fontsize='8', color='gray')
                    
                    for _, rule in top_rules.iterrows():
                        ant = rule['antecedents']
                        con = rule['consequents']
                        # Pendekkan nama jika terlalu panjang
                        ant_short = (ant[:20] + '..') if len(ant) > 20 else ant
                        con_short = (con[:20] + '..') if len(con) > 20 else con
                        
                        lift_str = f"Lift: {rule['lift']:.1f}x"
                        graph.edge(ant_short, con_short, label=lift_str)
                    
                    st.graphviz_chart(graph)
                    st.caption("Panah menunjuk ke produk yang direkomendasikan. (Jika beli A -> Tawarkan B).")

                with col_rules_list:
                    st.markdown("#### 📋 Top Paket Bundling")
                    # Loop 3 rule teratas
                    for i, row in top_rules.head(3).iterrows():
                        with st.container(border=True):
                            st.write(f"**Paket #{i+1}**")
                            st.write(f"📦 Beli: **{row['antecedents']}**")
                            st.write(f"➕ Tawarkan: **{row['consequents']}**")
                            
                            # Logika Strategi
                            if row['lift'] > 3:
                                st.caption("🔥 **Hard Bundle:** Jadikan satu paket fisik (diskon paket).")
                            elif row['lift'] > 1.5:
                                st.caption("📢 **Soft Bundle:** Rekomendasi 'Sering dibeli bersama' di web.")
                            else:
                                st.caption("👀 **Cross-Sell:** Letakkan bersebelahan.")
            else:
                st.warning("⚠️ Data Association Rules tidak cukup kuat atau belum tergenerate. Coba sesuaikan min_support pada kode.")
    else:
        # =============================================
        # LANDING PAGE (DENGAN TEKS YANG SUDAH DIEDIT)
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
        * **Pembobotan Langsung (Direct Weighting)**: Menentukan tingkat kepentingan kriteria secara manual via Slider.
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
        2. Atur **Konfigurasi Bobot** (Revenue, Sales, dll.) sesuai prioritas bisnis Anda di sidebar.
        3. Jelajahi hasil analisis di tab **"Rekomendasi TOPSIS"** untuk produk terbaik dan saran bundling.
        
        ---
        
        💡 **Tip**: Gunakan tab **"Analisis Detail Produk"** untuk menginspeksi produk secara spesifik.
        """)
        
        # Tampilkan preview metodologi
        st.markdown("---")
        st.subheader("🛠️ Arsitektur Sistem")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔍 Data Mining Workflow")
            st.graphviz_chart("""
            digraph {
                rankdir=TB;
                node [shape=box, style="filled,rounded", fillcolor="#e3f2fd", fontname="Arial"];
                edge [color="#90caf9"];
                
                A [label="Data Load & Clean"];
                B [label="Feature Engineering\n(RFM-like)"];
                C [label="K-Means Clustering\n(Segmentation)"];
                D [label="Apriori\n(Market Basket Analysis)"];
                E [label="Insight Generation\n(Rekomendasi)"];

                A -> B;
                B -> C;
                A -> D;
                C -> E;
                D -> E;
            }
            """)
        
        with col2:
            st.markdown("### 🎯 DSS Workflow")
            st.graphviz_chart("""
            digraph {
                rankdir=TB;
                node [shape=box, style="filled,rounded", fillcolor="#fff9c4", fontname="Arial"];
                edge [color="#ffe082"];
                
                A [label="Product Features\n(Data Produk)"];
                B [label="Bobot Kriteria\n(Slider User)"];
                C [label="TOPSIS Calculation\n(Perhitungan Jarak)"];
                D [label="Final Product\nRanking"];

                A -> C;
                B -> C;
                C -> D;
            }
            """)

if __name__ == "__main__":
    main()