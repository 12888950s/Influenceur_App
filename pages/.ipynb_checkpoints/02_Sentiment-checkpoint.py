import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# ─── Résolution du chemin racine du projet ───────────────────────────────────
# pages/01_Vue_Globale.py  →  parent = pages/  →  parent.parent = project/
_HERE = os.path.dirname(os.path.abspath(__file__))   # .../project/pages
_ROOT = os.path.dirname(_HERE)                        # .../project
_DATA = os.path.join(_ROOT, "data")                   # .../project/data

def _data(filename: str) -> str:
    """Retourne le chemin absolu vers un fichier dans data/."""
    return os.path.join(_DATA, filename)
st.set_page_config(
    page_title="Sentiments Influenceurs Tunisiens",
    page_icon="🇹🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Sora:wght@400;600;700;800&display=swap');

  /* ── Variables ── */
  :root {
    --dark-bg: #250e2c;
    --violet-mid: #837ab6;
    --violet-tender: #9d85b6;
    --accent-mauve: #cc8db3;
    --cta-pink: #f6a5c0;
    --bg-light: #f7c2ca;
    --white: #FFFFFF;
    --text-light: #f0e6f6;
    --text-muted: #c4b5fd;
    --glass-bg: rgba(255, 255, 255, 0.1);
    --glass-border: rgba(255, 255, 255, 0.18);
    --shadow-sm: 0 4px 15px rgba(0, 0, 0, 0.25);
    --shadow-md: 0 8px 30px rgba(0, 0, 0, 0.3);
    --radius-sm: 12px;
    --radius-md: 16px;
    --radius-lg: 24px;
  }

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text-light);
  }
  h1, h2, h3, h4, h5, h6 {
    font-family: 'Sora', sans-serif !important;
    color: var(--white) !important;
  }

  /* ── Fond général — Dégradé sombre élégant ── */
  .stApp {
    background: linear-gradient(
      160deg,
      #1a0a24 0%,
      #250e2c 25%,
      #2d1040 50%,
      #1a0a24 100%
    ) !important;
    background-attachment: fixed;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #3d1a4a 0%, #2d1040 50%, #1a0a24 100%) !important;
    border-right: 1px solid rgba(246, 165, 192, 0.3) !important;
  }
  [data-testid="stSidebar"] * { color: var(--text-light) !important; }
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 {
    color: var(--white) !important;
    font-family: 'Sora', sans-serif !important;
  }
  [data-testid="stSidebar"] hr {
    border-color: rgba(246, 165, 192, 0.3) !important;
  }

  /* ── Metric cards — Glassmorphism ── */
  [data-testid="stMetric"] {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--radius-md) !important;
    padding: 20px 24px !important;
    box-shadow: var(--shadow-sm) !important;
    transition: transform 0.25s, box-shadow 0.25s !important;
  }
  [data-testid="stMetric"]:hover {
    transform: translateY(-4px) !important;
    box-shadow: 0 8px 30px rgba(246, 165, 192, 0.3) !important;
    border-color: rgba(246, 165, 192, 0.4) !important;
  }
  [data-testid="stMetricLabel"] {
    color: var(--cta-pink) !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
  }
  [data-testid="stMetricValue"] {
    color: var(--white) !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 30px !important;
    font-weight: 800 !important;
  }
  [data-testid="stMetricDelta"] { font-size: 12px !important; }

  /* ── Tabs ── */
  [data-testid="stTabs"] {
    background: var(--glass-bg);
    border-radius: var(--radius-md);
    padding: 6px;
    border: 1px solid var(--glass-border);
    box-shadow: var(--shadow-sm);
    margin-bottom: 10px;
  }
  [data-testid="stTabs"] button {
    background: transparent !important;
    color: var(--text-muted) !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    border: none !important;
    transition: all 0.25s !important;
    padding: 8px 16px !important;
  }
  [data-testid="stTabs"] button:hover {
    background: rgba(246, 165, 192, 0.2) !important;
    color: var(--white) !important;
  }
  [data-testid="stTabs"] button[aria-selected="true"] {
    background: linear-gradient(135deg, var(--cta-pink), var(--accent-mauve)) !important;
    color: var(--dark-bg) !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 14px rgba(246, 165, 192, 0.4) !important;
  }

  /* ── DataTable ── */
  [data-testid="stDataFrame"] {
    border-radius: var(--radius-md);
    overflow: hidden;
    border: 1px solid var(--glass-border) !important;
    box-shadow: var(--shadow-sm);
    background: var(--glass-bg);
    backdrop-filter: blur(8px);
  }
  [data-testid="stDataFrame"] * { color: var(--text-light) !important; }

  /* ── Score badge ── */
  .score-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    font-family: 'Sora', sans-serif;
    letter-spacing: 0.02em;
  }

  /* ── Hero header ── */
  .hero {
    background: linear-gradient(135deg, #f6a5c0 0%, #cc8db3 35%, #837ab6 100%) !important;
    border-radius: var(--radius-lg);
    padding: 36px 44px;
    margin-bottom: 28px;
    box-shadow: 0 15px 50px rgba(246, 165, 192, 0.35);
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 280px; height: 280px;
    background: rgba(255, 255, 255, 0.2);
    border-radius: 50%;
  }
  .hero::after {
    content: '';
    position: absolute;
    bottom: -50px; left: 20%;
    width: 200px; height: 200px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 50%;
  }
  .hero h1 {
    margin: 0 0 8px 0 !important;
    font-size: 28px !important;
    color: var(--white) !important;
    font-weight: 800 !important;
    position: relative;
    z-index: 1;
    text-shadow: 0 2px 10px rgba(37, 14, 44, 0.3);
  }
  .hero p {
    margin: 0;
    color: rgba(255, 255, 255, 0.95) !important;
    font-size: 14px;
    position: relative;
    z-index: 1;
  }

  /* ── Influencer card — Glassmorphism ── */
  .inf-card {
    background: var(--glass-bg);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-md);
    padding: 16px 20px;
    margin-bottom: 10px;
    box-shadow: var(--shadow-sm);
    transition: all 0.3s ease;
  }
  .inf-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 30px rgba(246, 165, 192, 0.25);
    border-color: rgba(246, 165, 192, 0.5);
  }
  .inf-name {
    font-family: 'Sora', sans-serif;
    font-size: 14px;
    font-weight: 700;
    color: var(--white);
    margin-bottom: 10px;
  }
  .bar-wrap {
    background: rgba(255, 255, 255, 0.12);
    border-radius: 6px;
    height: 7px;
    overflow: hidden;
    margin: 3px 0;
  }
  .bar-pos  { background: linear-gradient(90deg, #10b981, #34d399); height: 100%; border-radius: 6px; }
  .bar-neg  { background: linear-gradient(90deg, #ef4444, #f87171); height: 100%; border-radius: 6px; }
  .bar-neu  { background: linear-gradient(90deg, #837ab6, #9d85b6); height: 100%; border-radius: 6px; }
  .pct-row  { display:flex; justify-content:space-between; font-size:11px; color:var(--text-muted); margin-top:4px; }

  /* ── Inputs ── */
  [data-testid="stSelectbox"] > div,
  [data-testid="stMultiSelect"] > div {
    background: rgba(255, 255, 255, 0.1) !important;
    border: 1.5px solid var(--glass-border) !important;
    border-radius: 10px !important;
    color: var(--white) !important;
    box-shadow: var(--shadow-sm) !important;
  }
  [data-testid="stSelectbox"] > div:focus-within,
  [data-testid="stMultiSelect"] > div:focus-within {
    border-color: var(--cta-pink) !important;
    box-shadow: 0 0 0 3px rgba(246, 165, 192, 0.2) !important;
  }
  [data-testid="stTextInput"] input {
    background: rgba(255, 255, 255, 0.1) !important;
    border: 1.5px solid var(--glass-border) !important;
    border-radius: 10px !important;
    color: var(--white) !important;
    box-shadow: var(--shadow-sm) !important;
  }
  [data-testid="stTextInput"] input:focus {
    border-color: var(--cta-pink) !important;
    box-shadow: 0 0 0 3px rgba(246, 165, 192, 0.2) !important;
  }
  .stSlider [data-baseweb="slider"] { margin-top: 8px; }

  /* ── Séparateur ── */
  hr { border-color: rgba(246, 165, 192, 0.3) !important; }

  /* ── Info box ── */
  .stAlert {
    background: rgba(246, 165, 192, 0.15) !important;
    border: 1.5px solid rgba(246, 165, 192, 0.3) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--white) !important;
    backdrop-filter: blur(8px);
  }

  /* ── Stat row card ── */
  .stat-row {
    background: var(--glass-bg);
    backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-md);
    padding: 14px 18px;
    font-size: 13px;
    box-shadow: var(--shadow-sm);
  }
  .stat-row-item {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  }
  .stat-row-item:last-child { border-bottom: none; }
  .stat-label { color: var(--text-muted); font-weight: 500; }
  .stat-value { color: var(--white); font-weight: 700; font-family: 'Sora', sans-serif; }

  /* ── Download button ── */
  .stDownloadButton button {
    background: linear-gradient(135deg, var(--violet-mid), var(--violet-tender)) !important;
    color: var(--white) !important;
    border: none !important;
    border-radius: 30px !important;
    font-weight: 700 !important;
    padding: 10px 24px !important;
    box-shadow: 0 5px 18px rgba(131, 122, 182, 0.3) !important;
    transition: all 0.3s !important;
  }
  .stDownloadButton button:hover {
    box-shadow: 0 8px 25px rgba(131, 122, 182, 0.5) !important;
    transform: translateY(-2px) !important;
  }

  /* ── Sidebar footer ── */
  .sidebar-footer {
    font-size: 11px;
    color: var(--cta-pink);
    line-height: 1.8;
    padding: 12px 14px;
    background: rgba(246, 165, 192, 0.08);
    border-radius: var(--radius-sm);
    border: 1px solid rgba(246, 165, 192, 0.2);
  }

  /* ── Textes généraux ── */
  p, span, li { color: var(--text-light) !important; }
  strong { color: var(--white) !important; }
  .stMarkdown p { color: var(--text-light) !important; }
  .stCaption { color: var(--text-muted) !important; }
</style>
""", unsafe_allow_html=True)

# ─── Chargement des données ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    # Cherche d'abord dans data/, puis à la racine comme fallback
    candidates = [
        _data("user_sentiment_global_.csv"),
        os.path.join(_ROOT, "user_sentiment_global_.csv"),
    ]
    csv_path = next((p for p in candidates if os.path.exists(p)), None)
    if csv_path is None:
        st.error(
            "❌ Fichier introuvable : `user_sentiment_global_.csv`\n\n"
            f"Chemin cherché : `{candidates[0]}`\n\n"
            "Vérifie que le fichier est bien dans le dossier `data/`."
        )
        st.stop()
    df = pd.read_csv(csv_path)
    df["user_name"] = df["user_name"].str.strip()
    df["platform"] = df.apply(
        lambda r: "YouTube + TikTok" if r["youtube_total_comments"] > 0 and r["tiktok_total_comments"] > 0
        else ("YouTube" if r["youtube_total_comments"] > 0 else "TikTok"),
        axis=1,
    )
    df["category"] = pd.cut(
        df["global_score"],
        bins=[-100, 0, 40, 60, 80, 101],
        labels=["Négatif 🔴", "Faible 🟠", "Moyen 🟡", "Bon 🟢", "Excellent ⭐"],
    )
    df["pos_pct"] = df.apply(
        lambda r: r["youtube_positive_pct"] if r["youtube_total_comments"] > 0 else r["tiktok_positive_pct"], axis=1
    )
    df["neg_pct"] = df.apply(
        lambda r: r["youtube_negative_pct"] if r["youtube_total_comments"] > 0 else r["tiktok_negative_pct"], axis=1
    )
    df["neu_pct"] = df.apply(
        lambda r: r["youtube_neutral_pct"] if r["youtube_total_comments"] > 0 else r["tiktok_neutral_pct"], axis=1
    )
    df["total_comments"] = df["youtube_total_comments"].fillna(0) + df["tiktok_total_comments"].fillna(0)
    return df


df = load_data()

# ─── Sidebar filtres ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Filtres")
    st.markdown("---")

    platform_filter = st.multiselect(
        "Plateforme",
        options=["YouTube", "TikTok", "YouTube + TikTok"],
        default=["YouTube", "TikTok", "YouTube + TikTok"],
    )

    score_range = st.slider(
        "Score global (min – max)",
        min_value=int(df["global_score"].min()),
        max_value=100,
        value=(0, 100),
        step=1,
    )

    cat_filter = st.multiselect(
        "Catégorie",
        options=["Négatif 🔴", "Faible 🟠", "Moyen 🟡", "Bon 🟢", "Excellent ⭐"],
        default=["Négatif 🔴", "Faible 🟠", "Moyen 🟡", "Bon 🟢", "Excellent ⭐"],
    )

    search_query = st.text_input("🔎 Rechercher un influenceur", placeholder="Tapez un nom...")

    st.markdown("---")
    sort_by = st.selectbox(
        "Trier par",
        ["Score global ↓", "Score global ↑", "% Positif ↓", "% Négatif ↓", "Nom A-Z"],
    )
    st.markdown("---")
    st.markdown(
        "<div class='sidebar-footer'>📊 Analyse des sentiments<br>des followers d'influenceurs<br>tunisiens · 2024–2025</div>",
        unsafe_allow_html=True,
    )

# ─── Filtrage ────────────────────────────────────────────────────────────────
filtered = df[
    (df["platform"].isin(platform_filter))
    & (df["global_score"] >= score_range[0])
    & (df["global_score"] <= score_range[1])
    & (df["category"].isin(cat_filter))
]
if search_query:
    filtered = filtered[filtered["user_name"].str.contains(search_query, case=False, na=False)]

sort_map = {
    "Score global ↓": ("global_score", False),
    "Score global ↑": ("global_score", True),
    "% Positif ↓": ("pos_pct", False),
    "% Négatif ↓": ("neg_pct", False),
    "Nom A-Z": ("user_name", True),
}
sc, asc = sort_map[sort_by]
filtered = filtered.sort_values(sc, ascending=asc).reset_index(drop=True)

# ─── Hero header ─────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="hero">
      <h1>🇹🇳 Sentiments des Followers · Influenceurs Tunisiens</h1>
      <p>Analyse de {len(df)} influenceurs · YouTube & TikTok · <strong style="color:rgba(255,255,255,0.9)">{filtered.shape[0]} résultats filtrés</strong></p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── KPI cards ───────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("📋 Influenceurs", len(filtered))
k2.metric("📈 Score moyen", f"{filtered['global_score'].mean():.1f}")
k3.metric("🏆 Score max", f"{filtered['global_score'].max():.1f}")
k4.metric("💚 % Positif moy.", f"{filtered['pos_pct'].mean():.1f}%")
k5.metric("❤️ % Négatif moy.", f"{filtered['neg_pct'].mean():.1f}%")

st.markdown("<br>", unsafe_allow_html=True)

# ─── Plotly theme helper ──────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.05)",
    font_color="#c4b5fd",
    title_font_color="#f0e6f6",
    title_font_family="Sora",
    title_font_size=14,
)
GRID = dict(gridcolor="rgba(255,255,255,0.08)", linecolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.08)")

# ─── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📋  Liste & Cartes", "📊  Analyses Visuelles", "🏆  Classement", "📥  Export"]
)

# ══════════════════════ TAB 1 : LISTE & CARTES ════════════════════════════════
with tab1:
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown(f"#### {len(filtered)} influenceurs trouvés")

        for _, row in filtered.iterrows():
            score = row["global_score"]
            if score >= 80:
                badge_style = "background:linear-gradient(135deg,#d1fae5,#a7f3d0);color:#065f46;border:1px solid #6ee7b7"
                emoji = "⭐"
            elif score >= 60:
                badge_style = "background:linear-gradient(135deg,#dbeafe,#bfdbfe);color:#1e40af;border:1px solid #93c5fd"
                emoji = "🟢"
            elif score >= 40:
                badge_style = "background:linear-gradient(135deg,#fef9c3,#fef08a);color:#854d0e;border:1px solid #fcd34d"
                emoji = "🟡"
            elif score >= 0:
                badge_style = "background:linear-gradient(135deg,#ffedd5,#fed7aa);color:#9a3412;border:1px solid #fdba74"
                emoji = "🟠"
            else:
                badge_style = "background:linear-gradient(135deg,#fee2e2,#fecaca);color:#991b1b;border:1px solid #f87171"
                emoji = "🔴"

            platform_icon = "▶️" if row["platform"] == "YouTube" else ("🎵" if row["platform"] == "TikTok" else "▶️🎵")
            pos = min(row["pos_pct"], 100)
            neg = min(row["neg_pct"], 100)
            neu = min(row["neu_pct"], 100)

            st.markdown(
                f"""
                <div class="inf-card">
                  <div style="display:flex;justify-content:space-between;align-items:center">
                    <div class="inf-name">{platform_icon} {row['user_name']}</div>
                    <span class="score-badge" style="{badge_style}">{emoji} {score:.1f}</span>
                  </div>
                  <div style="display:flex;gap:14px;margin-top:10px">
                    <div style="flex:1">
                      <div style="font-size:10px;color:#10b981;font-weight:700;text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px">Positif</div>
                      <div class="bar-wrap"><div class="bar-pos" style="width:{pos}%"></div></div>
                      <div style="font-size:11px;color:#c4b5fd;margin-top:2px;font-weight:600">{pos:.1f}%</div>
                    </div>
                    <div style="flex:1">
                      <div style="font-size:10px;color:#9d85b6;font-weight:700;text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px">Neutre</div>
                      <div class="bar-wrap"><div class="bar-neu" style="width:{neu}%"></div></div>
                      <div style="font-size:11px;color:#c4b5fd;margin-top:2px;font-weight:600">{neu:.1f}%</div>
                    </div>
                    <div style="flex:1">
                      <div style="font-size:10px;color:#ef4444;font-weight:700;text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px">Négatif</div>
                      <div class="bar-wrap"><div class="bar-neg" style="width:{neg}%"></div></div>
                      <div style="font-size:11px;color:#c4b5fd;margin-top:2px;font-weight:600">{neg:.1f}%</div>
                    </div>
                  </div>
                  <div style="font-size:11px;color:rgba(196,181,253,0.8);margin-top:8px;display:flex;align-items:center;gap:6px">
                    <span>💬 {int(row['total_comments'])} commentaires</span>
                    <span style="color:rgba(255,255,255,0.25)">·</span>
                    <span>{row['platform']}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_right:
        st.markdown("#### 🔎 Détail d'un influenceur")
        selected_name = st.selectbox(
            "Choisir un influenceur",
            options=filtered["user_name"].tolist(),
            label_visibility="collapsed",
        )
        row = filtered[filtered["user_name"] == selected_name].iloc[0]

        has_yt = row["youtube_total_comments"] > 0
        has_tt = row["tiktok_total_comments"] > 0

        def make_donut(pos, neg, neu, title, score_val):
            fig = go.Figure(
                go.Pie(
                    values=[pos, neg, neu],
                    labels=["Positif", "Négatif", "Neutre"],
                    hole=0.65,
                    marker_colors=["#10b981", "#ef4444", "#94a3b8"],
                    textinfo="percent",
                    textfont=dict(size=12, color="#2d1b69"),
                    marker=dict(line=dict(color="#ffffff", width=3)),
                )
            )
            fig.update_layout(
                title=dict(text=title, font=dict(color="#f0e6f6", size=13, family="Sora")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(font=dict(color="#c4b5fd", size=11), bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=10, r=10, t=40, b=10),
                annotations=[
                    dict(
                        text=f"<b>{score_val}</b>",
                        x=0.5, y=0.5,
                        font_size=20,
                        font_color="#ffffff",
                        showarrow=False,
                    )
                ],
                height=240,
            )
            return fig

        score_display = f"{row['global_score']:.0f}"

        if has_yt:
            fig_yt = make_donut(
                row["youtube_positive_pct"],
                row["youtube_negative_pct"],
                row["youtube_neutral_pct"],
                "▶️ YouTube",
                score_display,
            )
            st.plotly_chart(fig_yt, use_container_width=True, config={"displayModeBar": False})

        if has_tt:
            fig_tt = make_donut(
                row["tiktok_positive_pct"],
                row["tiktok_negative_pct"],
                row["tiktok_neutral_pct"],
                "🎵 TikTok",
                score_display,
            )
            st.plotly_chart(fig_tt, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            f"""
            <div class="stat-row">
              <div class="stat-row-item">
                <span class="stat-label">📊 Score global</span>
                <span class="stat-value">{row['global_score']:.2f}</span>
              </div>
              <div class="stat-row-item">
                <span class="stat-label">🏷️ Catégorie</span>
                <span class="stat-value">{row['category']}</span>
              </div>
              <div class="stat-row-item">
                <span class="stat-label">📱 Plateforme</span>
                <span class="stat-value">{row['platform']}</span>
              </div>
              <div class="stat-row-item">
                <span class="stat-label">💬 Commentaires</span>
                <span class="stat-value">{int(row['total_comments']):,}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ══════════════════════ TAB 2 : ANALYSES VISUELLES ════════════════════════════
with tab2:
    c1, c2 = st.columns(2)

    with c1:
        fig_hist = px.histogram(
            filtered,
            x="global_score",
            nbins=20,
            color_discrete_sequence=["#cc8db3"],
            title="Distribution des scores globaux",
            labels={"global_score": "Score global", "count": "Nombre"},
        )
        fig_hist.update_traces(marker_line_color="#250e2c", marker_line_width=1.5, opacity=0.85)
        fig_hist.update_layout(
            **PLOT_LAYOUT,
            xaxis=dict(**GRID, color="#c4b5fd"),
            yaxis=dict(**GRID, color="#c4b5fd"),
            bargap=0.06,
            height=320,
        )
        st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

    with c2:
        cat_counts = filtered["category"].value_counts().reset_index()
        cat_counts.columns = ["Catégorie", "Nombre"]
        fig_pie = px.pie(
            cat_counts,
            names="Catégorie",
            values="Nombre",
            hole=0.55,
            title="Répartition par catégorie",
            color_discrete_sequence=["#ef4444", "#f97316", "#eab308", "#10b981", "#7c3aed"],
        )
        fig_pie.update_traces(marker=dict(line=dict(color="#ffffff", width=3)))
        fig_pie.update_layout(
            **PLOT_LAYOUT,
            legend=dict(font=dict(color="#c4b5fd")),
            height=320,
        )
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

    fig_scatter = px.scatter(
        filtered,
        x="pos_pct",
        y="neg_pct",
        size="total_comments",
        color="global_score",
        hover_name="user_name",
        color_continuous_scale="RdYlGn",
        title="% Positif vs % Négatif  (taille = nb commentaires)",
        labels={"pos_pct": "% Positif", "neg_pct": "% Négatif", "global_score": "Score"},
        size_max=30,
    )
    fig_scatter.update_layout(
        **PLOT_LAYOUT,
        xaxis=dict(**GRID),
        yaxis=dict(**GRID),
        height=380,
    )
    st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})

    fig_box = px.box(
        filtered,
        x="category",
        y="global_score",
        color="category",
        title="Distribution des scores par catégorie",
        color_discrete_sequence=["#ef4444", "#f97316", "#eab308", "#10b981", "#7c3aed"],
        labels={"global_score": "Score global", "category": "Catégorie"},
    )
    fig_box.update_layout(
        **PLOT_LAYOUT,
        xaxis=dict(**GRID),
        yaxis=dict(**GRID),
        showlegend=False,
        height=340,
    )
    st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})

    plat_avg = (
        filtered.groupby("platform")[["pos_pct", "neg_pct", "neu_pct"]]
        .mean()
        .reset_index()
        .melt(id_vars="platform", var_name="Sentiment", value_name="Moyenne %")
    )
    plat_avg["Sentiment"] = plat_avg["Sentiment"].map(
        {"pos_pct": "Positif", "neg_pct": "Négatif", "neu_pct": "Neutre"}
    )
    fig_plat = px.bar(
        plat_avg,
        x="platform",
        y="Moyenne %",
        color="Sentiment",
        barmode="group",
        title="Sentiment moyen par plateforme",
        color_discrete_map={"Positif": "#10b981", "Négatif": "#ef4444", "Neutre": "#94a3b8"},
    )
    fig_plat.update_traces(marker_line_color="#ffffff", marker_line_width=1.5)
    fig_plat.update_layout(
        **PLOT_LAYOUT,
        xaxis=dict(**GRID),
        yaxis=dict(**GRID),
        legend=dict(font=dict(color="#c4b5fd"), bgcolor="rgba(0,0,0,0)"),
        height=320,
    )
    st.plotly_chart(fig_plat, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════ TAB 3 : CLASSEMENT ═══════════════════════════════════
with tab3:
    st.markdown("#### 🏆 Top 20 — Meilleurs scores")
    top20 = filtered.nlargest(20, "global_score").reset_index(drop=True)
    top20.index += 1

    fig_top = px.bar(
        top20,
        y="user_name",
        x="global_score",
        orientation="h",
        color="global_score",
        color_continuous_scale="RdYlGn",
        text="global_score",
        labels={"global_score": "Score global", "user_name": ""},
        height=560,
    )
    fig_top.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside",
        textfont_color="#f0e6f6",
        textfont_size=11,
        marker_line_color="#250e2c",
        marker_line_width=1,
    )
    fig_top.update_layout(
        **PLOT_LAYOUT,
        coloraxis_showscale=False,
        xaxis=dict(**GRID),
        yaxis=dict(autorange="reversed", color="#c4b5fd"),
        margin=dict(l=140),
    )
    st.plotly_chart(fig_top, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")
    st.markdown("#### 📉 Bottom 10 — Scores les plus bas")
    bot10 = filtered.nsmallest(10, "global_score").reset_index(drop=True)
    bot10.index += 1

    fig_bot = px.bar(
        bot10,
        y="user_name",
        x="global_score",
        orientation="h",
        color="global_score",
        color_continuous_scale="RdYlGn",
        text="global_score",
        labels={"global_score": "Score global", "user_name": ""},
        height=300,
    )
    fig_bot.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside",
        textfont_color="#f0e6f6",
        textfont_size=11,
        marker_line_color="#250e2c",
        marker_line_width=1,
    )
    fig_bot.update_layout(
        **PLOT_LAYOUT,
        coloraxis_showscale=False,
        xaxis=dict(**GRID),
        yaxis=dict(autorange="reversed", color="#c4b5fd"),
        margin=dict(l=140),
    )
    st.plotly_chart(fig_bot, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")
    st.markdown("#### 📊 Tableau complet")
    display_cols = ["user_name", "platform", "global_score", "pos_pct", "neg_pct", "neu_pct", "category", "total_comments"]
    col_rename = {
        "user_name": "Influenceur",
        "platform": "Plateforme",
        "global_score": "Score Global",
        "pos_pct": "% Positif",
        "neg_pct": "% Négatif",
        "neu_pct": "% Neutre",
        "category": "Catégorie",
        "total_comments": "Commentaires",
    }
    st.dataframe(
        filtered[display_cols].rename(columns=col_rename),
        use_container_width=True,
        height=400,
        column_config={
            "Score Global": st.column_config.ProgressColumn("Score Global", min_value=0, max_value=100, format="%.1f"),
            "% Positif": st.column_config.ProgressColumn("% Positif", min_value=0, max_value=100, format="%.1f%%"),
            "% Négatif": st.column_config.ProgressColumn("% Négatif", min_value=0, max_value=100, format="%.1f%%"),
        },
    )


# ══════════════════════ TAB 4 : EXPORT ═══════════════════════════════════════
with tab4:
    st.markdown("#### 📥 Exporter les données filtrées")
    st.info(f"**{len(filtered)} influenceurs** correspondent aux filtres actuels.")

    export_df = filtered[
        ["user_name", "platform", "global_score", "pos_pct", "neg_pct", "neu_pct", "category",
         "total_comments", "youtube_sentiment_score", "tiktok_sentiment_score"]
    ].rename(columns={
        "user_name": "Influenceur", "platform": "Plateforme", "global_score": "Score Global",
        "pos_pct": "% Positif", "neg_pct": "% Négatif", "neu_pct": "% Neutre",
        "category": "Catégorie", "total_comments": "Total Commentaires",
        "youtube_sentiment_score": "Score YouTube", "tiktok_sentiment_score": "Score TikTok",
    })

    csv_data = export_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="⬇️  Télécharger CSV",
        data=csv_data,
        file_name="sentiments_influenceurs_tunisiens.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.markdown("#### 📈 Statistiques de synthèse")
    stats = filtered.groupby("category").agg(
        Nombre=("user_name", "count"),
        Score_Moyen=("global_score", "mean"),
        Positif_Moy=("pos_pct", "mean"),
        Negatif_Moy=("neg_pct", "mean"),
    ).round(2).reset_index()
    stats.columns = ["Catégorie", "Nombre", "Score Moyen", "% Positif Moy.", "% Négatif Moy."]
    st.dataframe(stats, use_container_width=True, hide_index=True)