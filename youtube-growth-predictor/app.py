import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube Growth Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0a0a0f;
    color: #e8e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f0f1a !important;
    border-right: 1px solid #1e1e2e;
}

/* Header */
.main-header {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.5rem;
    letter-spacing: 3px;
    background: linear-gradient(135deg, #ff0000 0%, #ff6b35 50%, #ffcc00 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0;
    line-height: 1;
}

.sub-header {
    color: #666688;
    font-size: 0.95rem;
    font-weight: 300;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* Cards */
.metric-card {
    background: linear-gradient(135deg, #13131f 0%, #1a1a2e 100%);
    border: 1px solid #2a2a3e;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    transition: all 0.3s ease;
}

.metric-card:hover {
    border-color: #ff4444;
    transform: translateY(-2px);
}

.metric-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.2rem;
    color: #ff4444;
    letter-spacing: 2px;
}

.metric-label {
    color: #888;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* Prediction result */
.result-rising {
    background: linear-gradient(135deg, #0d2818, #1a4a2e);
    border: 2px solid #2ecc71;
    border-radius: 20px;
    padding: 32px;
    text-align: center;
}
.result-established {
    background: linear-gradient(135deg, #0d1828, #1a2e4a);
    border: 2px solid #3498db;
    border-radius: 20px;
    padding: 32px;
    text-align: center;
}
.result-dominant {
    background: linear-gradient(135deg, #28180d, #4a2e1a);
    border: 2px solid #f39c12;
    border-radius: 20px;
    padding: 32px;
    text-align: center;
}

.result-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.8rem;
    letter-spacing: 3px;
    margin: 0;
}

.result-emoji {
    font-size: 3.5rem;
    margin-bottom: 12px;
}

/* Inputs */
.stSlider > div > div > div {
    background: #ff4444 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #13131f;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #888;
    border-radius: 8px;
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    background: #ff4444 !important;
    color: white !important;
}

/* Divider */
.red-line {
    height: 3px;
    background: linear-gradient(90deg, #ff0000, #ff6b35, transparent);
    border: none;
    margin: 20px 0;
}

/* Section titles */
.section-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.6rem;
    letter-spacing: 2px;
    color: #ff4444;
    margin-bottom: 16px;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #ff0000, #ff6b35) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 40px !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.2rem !important;
    letter-spacing: 2px !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(255, 68, 68, 0.4) !important;
}

/* Info boxes */
.info-box {
    background: #13131f;
    border-left: 3px solid #ff4444;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin: 12px 0;
    font-size: 0.9rem;
    color: #aaa;
}

/* Number input */
[data-testid="stNumberInput"] input {
    background: #13131f !important;
    border: 1px solid #2a2a3e !important;
    color: #e8e8f0 !important;
    border-radius: 8px !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div {
    background: #13131f !important;
    border: 1px solid #2a2a3e !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Paths (works locally and on Streamlit Cloud) ──────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Model files can be inside a 'model' folder or right next to app.py
MODEL_DIR = os.path.join(BASE_DIR, 'model')
if not os.path.exists(os.path.join(MODEL_DIR, 'youtube_growth_model.pkl')):
    MODEL_DIR = BASE_DIR


# ─── Helper: Load Model ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model    = joblib.load(os.path.join(MODEL_DIR, 'youtube_growth_model.pkl'))
        le_cat   = joblib.load(os.path.join(MODEL_DIR, 'le_category.pkl'))
        le_chan  = joblib.load(os.path.join(MODEL_DIR, 'le_channel_type.pkl'))
        le_ctry  = joblib.load(os.path.join(MODEL_DIR, 'le_country.pkl'))
        return model, le_cat, le_chan, le_ctry, True
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None, None, None, None, False


@st.cache_data
def load_data():
    for name in ['Global_YouTube_Statistics.csv', 'global_youtube_statistics.csv']:
        path = os.path.join(BASE_DIR, name)
        if os.path.exists(path):
            return pd.read_csv(path, encoding='latin1')
    return None


def safe_encode(encoder, value):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    return 0


def predict(model, le_cat, le_chan, le_ctry,
            subscribers, video_views, uploads,
            views_30d, subs_30d,
            high_monthly, high_yearly,
            created_year, category, channel_type, country):

    views_per_sub   = video_views / (subscribers + 1)
    channel_age     = max(2024 - created_year, 0)
    uploads_per_yr  = uploads / (channel_age + 1)
    earnings_spread = high_monthly * 0.7
    view_growth     = views_30d / (video_views + 1)
    sub_growth      = subs_30d  / (subscribers + 1)

    input_df = pd.DataFrame([{
        'video views':                      video_views,
        'uploads':                          uploads,
        'video_views_for_the_last_30_days': views_30d,
        'subscribers_for_last_30_days':     subs_30d,
        'highest_monthly_earnings':         high_monthly,
        'highest_yearly_earnings':          high_yearly,
        'views_per_subscriber':             views_per_sub,
        'channel_age':                      channel_age,
        'uploads_per_year':                 uploads_per_yr,
        'earnings_spread':                  earnings_spread,
        'view_growth_rate':                 view_growth,
        'sub_growth_rate':                  sub_growth,
        'category_encoded':                 safe_encode(le_cat,  category),
        'channel_type_encoded':             safe_encode(le_chan,  channel_type),
        'country_encoded':                  safe_encode(le_ctry, country),
    }])

    prediction = model.predict(input_df)[0]
    proba      = model.predict_proba(input_df)[0]
    classes    = model.classes_
    return prediction, dict(zip(classes, proba))


# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px'>
        <span style='font-size:2.5rem'>🎬</span>
        <p style='font-family: Bebas Neue; font-size:1.3rem; color:#ff4444; letter-spacing:2px; margin:4px 0'>
        YOUTUBE GROWTH<br>PREDICTOR
        </p>
        <p style='color:#555; font-size:0.75rem; letter-spacing:1px'>ML PROJECT · GAT BENGALURU</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1e1e2e'>", unsafe_allow_html=True)
    st.markdown("<p style='color:#666; font-size:0.75rem; text-transform:uppercase; letter-spacing:1px'>Navigation</p>", unsafe_allow_html=True)

    page = st.radio("Navigation", ["🏠 Home", "🔮 Predict", "📊 Analytics", "ℹ️ About"],
                    label_visibility="collapsed")

    st.markdown("<hr style='border-color:#1e1e2e'>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
    <b style='color:#ff4444'>Model Info</b><br>
    Algorithm: Random Forest<br>
    Classes: 3 Growth Tiers<br>
    Features: 15 engineered
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='position:absolute; bottom:20px; left:0; right:0; text-align:center;'>
        <p style='color:#333; font-size:0.7rem'>CSE23602 · Machine Learning<br>Team: Prasad · Giri · Likhith · Manoj</p>
    </div>
    """, unsafe_allow_html=True)


# ─── HOME PAGE ─────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown("<div class='main-header'>YOUTUBE CREATOR<br>GROWTH PREDICTOR</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Machine Learning · Analytics Dashboard · Global Academy of Technology</div>", unsafe_allow_html=True)
    st.markdown("<div class='red-line'></div>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class='metric-card'>
            <div class='metric-value'>1,006</div>
            <div class='metric-label'>YouTube Channels</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class='metric-card'>
            <div class='metric-value'>29</div>
            <div class='metric-label'>Dataset Features</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class='metric-card'>
            <div class='metric-value'>3</div>
            <div class='metric-label'>Growth Tiers</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class='metric-card'>
            <div class='metric-value'>RF</div>
            <div class='metric-label'>Algorithm</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown("<div class='section-title'>ABOUT THIS PROJECT</div>", unsafe_allow_html=True)
        st.markdown("""
        <p style='color:#aaa; line-height:1.8'>
        This ML-powered dashboard predicts YouTube channel growth using the
        <b style='color:#ff4444'>Global YouTube Statistics 2023</b> dataset.
        Channels are classified into three growth tiers based on subscriber count,
        engagement metrics, and earning potential.
        </p>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-title' style='margin-top:24px'>GROWTH TIERS</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='display:flex; flex-direction:column; gap:10px'>
            <div style='background:#0d2818; border:1px solid #2ecc71; border-radius:12px; padding:14px 18px; display:flex; align-items:center; gap:12px'>
                <span style='font-size:1.5rem'>🌱</span>
                <div><b style='color:#2ecc71'>Rising Creator</b><br><span style='color:#666; font-size:0.85rem'>Bottom 33% — Growing channels with high potential</span></div>
            </div>
            <div style='background:#0d1828; border:1px solid #3498db; border-radius:12px; padding:14px 18px; display:flex; align-items:center; gap:12px'>
                <span style='font-size:1.5rem'>⭐</span>
                <div><b style='color:#3498db'>Established Creator</b><br><span style='color:#666; font-size:0.85rem'>Middle 33% — Stable, consistent performers</span></div>
            </div>
            <div style='background:#28180d; border:1px solid #f39c12; border-radius:12px; padding:14px 18px; display:flex; align-items:center; gap:12px'>
                <span style='font-size:1.5rem'>👑</span>
                <div><b style='color:#f39c12'>Dominant Creator</b><br><span style='color:#666; font-size:0.85rem'>Top 33% — Platform giants with massive reach</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-title'>ML PIPELINE</div>", unsafe_allow_html=True)
        steps = [
            ("01", "Data Collection", "Global YouTube Statistics 2023 (Kaggle)"),
            ("02", "Preprocessing", "Handle missing values, encode categories"),
            ("03", "Feature Engineering", "6 derived features: engagement, growth rates"),
            ("04", "Model Training", "Random Forest — 200 estimators"),
            ("05", "Evaluation", "Accuracy, Precision, Recall, F1-Score"),
            ("06", "Deployment", "Streamlit interactive dashboard"),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div style='display:flex; gap:16px; margin-bottom:14px; align-items:flex-start'>
                <div style='background:#ff4444; color:white; border-radius:8px; padding:4px 10px;
                            font-family:Bebas Neue; font-size:1rem; letter-spacing:1px; min-width:36px; text-align:center'>
                    {num}
                </div>
                <div>
                    <div style='color:#e8e8f0; font-weight:600; font-size:0.9rem'>{title}</div>
                    <div style='color:#666; font-size:0.8rem'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ─── PREDICT PAGE ──────────────────────────────────────────────────────────────
elif page == "🔮 Predict":
    st.markdown("<div class='main-header'>GROWTH PREDICTOR</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Enter your channel stats to predict growth tier</div>", unsafe_allow_html=True)
    st.markdown("<div class='red-line'></div>", unsafe_allow_html=True)

    model, le_cat, le_chan, le_ctry, model_loaded = load_model()

    if not model_loaded:
        st.markdown("""
        <div style='background:#1a1015; border:2px solid #ff4444; border-radius:16px; padding:24px; text-align:center'>
            <div style='font-size:2rem'>⚠️</div>
            <h3 style='color:#ff4444'>Model Not Found</h3>
            <p style='color:#888'>Please train the model first in Google Colab and place the <b>model/</b> folder in the same directory as app.py</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("<div class='section-title'>📺 CHANNEL INFO</div>", unsafe_allow_html=True)

            subscribers = st.number_input("Subscribers", min_value=0, value=1_000_000, step=100_000,
                                           help="Total subscriber count")
            video_views = st.number_input("Total Video Views", min_value=0, value=50_000_000, step=1_000_000)
            uploads     = st.number_input("Total Uploads", min_value=0, value=200, step=10)
            created_year = st.slider("Channel Created Year", 2005, 2024, 2015)

            categories = list(le_cat.classes_)
            category   = st.selectbox("Category", categories)

            channel_types = list(le_chan.classes_)
            channel_type  = st.selectbox("Channel Type", channel_types)

            countries = list(le_ctry.classes_)
            country   = st.selectbox("Country", countries,
                                      index=countries.index('India') if 'India' in countries else 0)

        with col2:
            st.markdown("<div class='section-title'>📈 GROWTH METRICS</div>", unsafe_allow_html=True)

            views_30d    = st.number_input("Views (Last 30 Days)", min_value=0, value=500_000, step=10_000)
            subs_30d     = st.number_input("New Subscribers (Last 30 Days)", min_value=0, value=10_000, step=1_000)
            high_monthly = st.number_input("Highest Monthly Earnings (USD)", min_value=0, value=5_000, step=500)
            high_yearly  = st.number_input("Highest Yearly Earnings (USD)", min_value=0, value=60_000, step=1_000)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class='info-box'>
            💡 <b style='color:#ff4444'>Tip:</b> You can find earnings estimates on Social Blade or YouTube analytics. 
            Use approximate values if exact data isn't available.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔮 PREDICT GROWTH TIER"):
            with st.spinner("Analyzing channel data..."):
                prediction, probas = predict(
                    model, le_cat, le_chan, le_ctry,
                    subscribers, video_views, uploads,
                    views_30d, subs_30d,
                    high_monthly, high_yearly,
                    created_year, category, channel_type, country
                )

            # Result display
            tier_config = {
                'Rising Creator':      ('result-rising',      '🌱', '#2ecc71'),
                'Established Creator': ('result-established', '⭐', '#3498db'),
                'Dominant Creator':    ('result-dominant',    '👑', '#f39c12'),
            }
            css_class, emoji, color = tier_config.get(prediction, ('result-rising', '🌱', '#2ecc71'))

            st.markdown(f"""
            <div class='{css_class}'>
                <div class='result-emoji'>{emoji}</div>
                <p class='result-title' style='color:{color}'>{str(prediction).upper()}</p>
                <p style='color:#888; margin-top:8px; font-size:0.9rem'>Based on your channel metrics</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>CONFIDENCE SCORES</div>", unsafe_allow_html=True)

            colors_map = {
                'Rising Creator': '#2ecc71',
                'Established Creator': '#3498db',
                'Dominant Creator': '#f39c12'
            }
            for tier, prob in sorted(probas.items(), key=lambda x: x[1], reverse=True):
                bar_color = colors_map.get(tier, '#ff4444')
                st.markdown(f"""
                <div style='margin-bottom:12px'>
                    <div style='display:flex; justify-content:space-between; margin-bottom:4px'>
                        <span style='color:#ccc; font-size:0.9rem'>{tier}</span>
                        <span style='color:{bar_color}; font-weight:600'>{prob*100:.1f}%</span>
                    </div>
                    <div style='background:#1a1a2e; border-radius:999px; height:8px; overflow:hidden'>
                        <div style='width:{prob*100}%; background:{bar_color}; height:100%; border-radius:999px;
                                    transition:width 1s ease'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ─── ANALYTICS PAGE ────────────────────────────────────────────────────────────
elif page == "📊 Analytics":
    st.markdown("<div class='main-header'>ANALYTICS DASHBOARD</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Explore YouTube creator trends and insights</div>", unsafe_allow_html=True)
    st.markdown("<div class='red-line'></div>", unsafe_allow_html=True)

    df = load_data()
    if df is None:
        st.markdown("""
        <div style='background:#1a1015; border:2px solid #ff4444; border-radius:16px; padding:24px; text-align:center'>
            <div style='font-size:2rem'>⚠️</div>
            <h3 style='color:#ff4444'>Dataset Not Found</h3>
            <p style='color:#888'>Place <b>Global_YouTube_Statistics.csv</b> in the same folder as app.py</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Clean for display
        df = df.copy()
        df['category'] = df['category'].fillna('Unknown')
        df['Country'] = df['Country'].fillna('Unknown')

        plt.style.use('dark_background')
        bg = '#13131f'
        accent = '#ff4444'

        tab1, tab2, tab3 = st.tabs(["🏆 Top Channels", "📂 Categories", "🌍 Countries"])

        with tab1:
            st.markdown("<div class='section-title'>TOP 10 CHANNELS BY SUBSCRIBERS</div>", unsafe_allow_html=True)
            top10 = df.nlargest(10, 'subscribers')[['Youtuber', 'subscribers', 'category', 'Country']]

            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor(bg)
            ax.set_facecolor(bg)
            bars = ax.barh(top10['Youtuber'], top10['subscribers'] / 1e6,
                           color=[accent] + ['#3a3a5a'] * 9)
            ax.set_xlabel('Subscribers (Millions)', color='#888')
            ax.tick_params(colors='#ccc')
            ax.spines[:].set_color('#2a2a3e')
            ax.invert_yaxis()
            st.pyplot(fig)
            plt.close()

        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<div class='section-title'>CHANNEL COUNT</div>", unsafe_allow_html=True)
                cat_counts = df['category'].value_counts().head(8)
                fig, ax = plt.subplots(figsize=(6, 5))
                fig.patch.set_facecolor(bg)
                ax.set_facecolor(bg)
                colors = [accent if i == 0 else '#3a3a5a' for i in range(len(cat_counts))]
                ax.barh(cat_counts.index, cat_counts.values, color=colors)
                ax.tick_params(colors='#ccc')
                ax.spines[:].set_color('#2a2a3e')
                ax.invert_yaxis()
                st.pyplot(fig)
                plt.close()

            with col2:
                st.markdown("<div class='section-title'>AVG MONTHLY EARNINGS</div>", unsafe_allow_html=True)
                avg_earn = df.groupby('category')['highest_monthly_earnings'].mean().sort_values(ascending=False).head(8)
                fig, ax = plt.subplots(figsize=(6, 5))
                fig.patch.set_facecolor(bg)
                ax.set_facecolor(bg)
                colors = [accent if i == 0 else '#3a3a5a' for i in range(len(avg_earn))]
                ax.barh(avg_earn.index, avg_earn.values, color=colors)
                ax.tick_params(colors='#ccc')
                ax.spines[:].set_color('#2a2a3e')
                ax.invert_yaxis()
                st.pyplot(fig)
                plt.close()

        with tab3:
            st.markdown("<div class='section-title'>TOP 10 COUNTRIES BY CHANNEL COUNT</div>", unsafe_allow_html=True)
            top_ctry = df['Country'].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor(bg)
            ax.set_facecolor(bg)
            colors = [accent if i == 0 else '#3a3a5a' for i in range(len(top_ctry))]
            ax.bar(top_ctry.index, top_ctry.values, color=colors)
            ax.tick_params(colors='#ccc', axis='x', rotation=30)
            ax.tick_params(colors='#ccc', axis='y')
            ax.spines[:].set_color('#2a2a3e')
            st.pyplot(fig)
            plt.close()


# ─── ABOUT PAGE ────────────────────────────────────────────────────────────────
elif page == "ℹ️ About":
    st.markdown("<div class='main-header'>ABOUT</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Project details and team information</div>", unsafe_allow_html=True)
    st.markdown("<div class='red-line'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("<div class='section-title'>PROJECT DETAILS</div>", unsafe_allow_html=True)
        details = [
            ("🏫 Institute",   "Global Academy of Technology, Bengaluru"),
            ("📚 Course",      "CSE23602 – Machine Learning"),
            ("👩‍🏫 Faculty",    "Prof. Yashaswini K"),
            ("📅 Submitted",   "04/05/2026"),
            ("🗄️ Dataset",     "Global YouTube Statistics 2023 (Kaggle)"),
            ("🤖 Algorithm",   "Random Forest Classifier"),
            ("🐍 Framework",   "Python · Scikit-learn · Streamlit"),
        ]
        for icon_label, value in details:
            st.markdown(f"""
            <div style='display:flex; gap:12px; margin-bottom:12px; background:#13131f;
                        border:1px solid #2a2a3e; border-radius:10px; padding:12px 16px'>
                <span style='color:#ff4444; font-weight:600; min-width:120px; font-size:0.85rem'>{icon_label}</span>
                <span style='color:#aaa; font-size:0.85rem'>{value}</span>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-title'>TEAM MEMBERS</div>", unsafe_allow_html=True)
        members = [
            ("1GA23CS123", "Prasad KS"),
            ("1GA23CS129", "RS Giri"),
            ("1GA24CS404", "Likhith D"),
            ("1GA24CS405", "Manoj DA"),
        ]
        for usn, name in members:
            st.markdown(f"""
            <div style='background:#13131f; border:1px solid #2a2a3e; border-radius:14px;
                        padding:18px 20px; margin-bottom:12px; display:flex; align-items:center; gap:16px'>
                <div style='background:linear-gradient(135deg,#ff0000,#ff6b35); border-radius:50%;
                            width:44px; height:44px; display:flex; align-items:center; justify-content:center;
                            font-family:Bebas Neue; font-size:1.1rem; color:white; flex-shrink:0'>
                    {name[0]}
                </div>
                <div>
                    <div style='color:#e8e8f0; font-weight:600'>{name}</div>
                    <div style='color:#555; font-size:0.8rem; letter-spacing:1px'>{usn}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='section-title' style='margin-top:20px'>REFERENCES</div>", unsafe_allow_html=True)
        refs = ["XGBoost — Chen & Guestrin (2016)", "Random Forests — Breiman (2001)",
                "SHAP — Lundberg & Lee (2017)", "Scikit-learn — Pedregosa et al. (2011)"]
        for r in refs:
            st.markdown(f"<div class='info-box'>📄 {r}</div>", unsafe_allow_html=True)
