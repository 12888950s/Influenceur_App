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
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Sora:wght@400;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
  }

  /* ── Fond général ── */
  .stApp {
    background: linear-gradient(160deg, #f0f4ff 0%, #faf8ff 50%, #f5f0ff 100%);
    color: #1a1a2e;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f8f5ff 100%) !important;
    border-right: 1px solid #e8e0ff !important;
    box-shadow: 4px 0 20px rgba(124, 58, 237, 0.06);
  }
  [data-testid="stSidebar"] * { color: #3d3467 !important; }
  [data-testid="stSidebar"] h3 {
    font-family: 'Sora', sans-serif !important;
    font-size: 16px !important;
    color: #5b21b6 !important;
    letter-spacing: 0.02em;
  }

  /* ── Metric cards ── */
  [data-testid="stMetric"] {
    background: #ffffff;
    border: 1.5px solid #ede9fe;
    border-radius: 16px;
    padding: 18px 22px !important;
    box-shadow: 0 4px 20px rgba(109, 40, 217, 0.07);
    transition: transform 0.2s, box-shadow 0.2s;
  }
  [data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(109, 40, 217, 0.13);
  }
  [data-testid="stMetricLabel"] {
    color: #7c6ea8 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  [data-testid="stMetricValue"] {
    color: #2d1b69 !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 28px !important;
    font-weight: 700 !important;
  }
  [data-testid="stMetricDelta"] { font-size: 12px !important; }

  /* ── Titres ── */
  h1, h2, h3, h4 {
    font-family: 'Sora', sans-serif !important;
    color: #2d1b69 !important;
  }

  /* ── Tabs ── */
  [data-testid="stTabs"] {
    background: #ffffff;
    border-radius: 14px;
    padding: 4px;
    border: 1.5px solid #ede9fe;
    box-shadow: 0 2px 10px rgba(109,40,217,0.06);
    margin-bottom: 8px;
  }
  [data-testid="stTabs"] button {
    background: transparent;
    color: #7c6ea8;
    border-radius: 10px;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 600;
    font-size: 13px;
    padding: 8px 16px;
    border: none !important;
    transition: all 0.2s;
  }
  [data-testid="stTabs"] button:hover {
    background: #f5f0ff;
    color: #5b21b6;
  }
  [data-testid="stTabs"] button[aria-selected="true"] {
    background: linear-gradient(135deg, #7c3aed 0%, #5b21b6 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 12px rgba(124, 58, 237, 0.35) !important;
  }

  /* ── DataTable ── */
  [data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1.5px solid #ede9fe !important;
    box-shadow: 0 4px 16px rgba(109,40,217,0.06);
  }

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
    background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 50%, #0ea5e9 100%);
    border-radius: 20px;
    padding: 32px 40px;
    margin-bottom: 28px;
    box-shadow: 0 12px 40px rgba(124, 58, 237, 0.28);
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: rgba(255,255,255,0.07);
    border-radius: 50%;
  }
  .hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 160px; height: 160px;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
  }
  .hero h1 {
    margin: 0 0 8px 0 !important;
    font-size: 26px !important;
    color: #ffffff !important;
    font-weight: 700 !important;
  }
  .hero p { margin: 0; color: rgba(255,255,255,0.75); font-size: 14px; }

  /* ── Influencer card ── */
  .inf-card {
    background: #ffffff;
    border: 1.5px solid #ede9fe;
    border-radius: 16px;
    padding: 16px 20px;
    margin-bottom: 10px;
    box-shadow: 0 2px 12px rgba(109,40,217,0.05);
    transition: border-color 0.2s, box-shadow 0.2s, transform 0.15s;
  }
  .inf-card:hover {
    border-color: #7c3aed;
    box-shadow: 0 6px 24px rgba(124, 58, 237, 0.14);
    transform: translateY(-1px);
  }
  .inf-name {
    font-family: 'Sora', sans-serif;
    font-size: 14px;
    font-weight: 700;
    color: #2d1b69;
    margin-bottom: 10px;
  }
  .bar-wrap {
    background: #f0ebff;
    border-radius: 6px;
    height: 7px;
    overflow: hidden;
    margin: 3px 0;
  }
  .bar-pos  { background: linear-gradient(90deg, #10b981, #34d399); height: 100%; border-radius: 6px; }
  .bar-neg  { background: linear-gradient(90deg, #ef4444, #f87171); height: 100%; border-radius: 6px; }
  .bar-neu  { background: linear-gradient(90deg, #94a3b8, #cbd5e1); height: 100%; border-radius: 6px; }
  .pct-row  { display:flex; justify-content:space-between; font-size:11px; color:#9381c0; margin-top:4px; }

  /* ── Inputs ── */
  [data-testid="stSelectbox"] > div,
  [data-testid="stMultiSelect"] > div {
    background: #ffffff !important;
    border: 1.5px solid #ddd6fe !important;
    border-radius: 10px !important;
    color: #3d3467 !important;
    box-shadow: 0 2px 8px rgba(109,40,217,0.06) !important;
  }
  [data-testid="stTextInput"] input {
    background: #ffffff !important;
    border: 1.5px solid #ddd6fe !important;
    border-radius: 10px !important;
    color: #3d3467 !important;
    box-shadow: 0 2px 8px rgba(109,40,217,0.06) !important;
  }
  .stSlider [data-baseweb="slider"] { margin-top: 8px; }

  /* ── Séparateur ── */
  hr { border-color: #ede9fe; }

  /* ── Info box ── */
  .stAlert {
    background: #f5f0ff !important;
    border: 1.5px solid #ddd6fe !important;
    border-radius: 12px !important;
    color: #5b21b6 !important;
  }

  /* ── Stat row card ── */
  .stat-row {
    background: #fafbff;
    border: 1.5px solid #ede9fe;
    border-radius: 14px;
    padding: 14px 18px;
    font-size: 13px;
  }
  .stat-row-item {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid #f0ebff;
  }
  .stat-row-item:last-child { border-bottom: none; }
  .stat-label { color: #9381c0; font-weight: 500; }
  .stat-value { color: #2d1b69; font-weight: 700; font-family: 'Sora', sans-serif; }

  /* ── Download button ── */
  .stDownloadButton button {
    background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    box-shadow: 0 4px 14px rgba(124,58,237,0.3) !important;
    transition: all 0.2s !important;
  }
  .stDownloadButton button:hover {
    box-shadow: 0 6px 20px rgba(124,58,237,0.45) !important;
    transform: translateY(-1px) !important;
  }

  /* ── Sidebar footer ── */
  .sidebar-footer {
    font-size: 11px;
    color: #9381c0;
    line-height: 1.6;
    padding: 10px 12px;
    background: #f5f0ff;
    border-radius: 10px;
    border: 1px solid #ede9fe;
  }
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
    plot_bgcolor="rgba(255,255,255,0.6)",
    font_color="#5b21b6",
    title_font_color="#2d1b69",
    title_font_family="Sora",
    title_font_size=14,
)
GRID = dict(gridcolor="#ede9fe", linecolor="#ede9fe", zerolinecolor="#ede9fe")

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
                      <div style="font-size:11px;color:#7c6ea8;margin-top:2px;font-weight:600">{pos:.1f}%</div>
                    </div>
                    <div style="flex:1">
                      <div style="font-size:10px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px">Neutre</div>
                      <div class="bar-wrap"><div class="bar-neu" style="width:{neu}%"></div></div>
                      <div style="font-size:11px;color:#7c6ea8;margin-top:2px;font-weight:600">{neu:.1f}%</div>
                    </div>
                    <div style="flex:1">
                      <div style="font-size:10px;color:#ef4444;font-weight:700;text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px">Négatif</div>
                      <div class="bar-wrap"><div class="bar-neg" style="width:{neg}%"></div></div>
                      <div style="font-size:11px;color:#7c6ea8;margin-top:2px;font-weight:600">{neg:.1f}%</div>
                    </div>
                  </div>
                  <div style="font-size:11px;color:#9381c0;margin-top:8px;display:flex;align-items:center;gap:6px">
                    <span>💬 {int(row['total_comments'])} commentaires</span>
                    <span style="color:#ddd6fe">·</span>
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
                title=dict(text=title, font=dict(color="#2d1b69", size=13, family="Sora")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(font=dict(color="#7c6ea8", size=11), bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=10, r=10, t=40, b=10),
                annotations=[
                    dict(
                        text=f"<b>{score_val}</b>",
                        x=0.5, y=0.5,
                        font_size=20,
                        font_color="#2d1b69",
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
            color_discrete_sequence=["#7c3aed"],
            title="Distribution des scores globaux",
            labels={"global_score": "Score global", "count": "Nombre"},
        )
        fig_hist.update_traces(marker_line_color="#ffffff", marker_line_width=1.5, opacity=0.85)
        fig_hist.update_layout(
            **PLOT_LAYOUT,
            xaxis=dict(**GRID, color="#7c6ea8"),
            yaxis=dict(**GRID, color="#7c6ea8"),
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
            legend=dict(font=dict(color="#5b21b6")),
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
        legend=dict(font=dict(color="#5b21b6"), bgcolor="rgba(0,0,0,0)"),
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
        textfont_color="#2d1b69",
        textfont_size=11,
        marker_line_color="#ffffff",
        marker_line_width=1,
    )
    fig_top.update_layout(
        **PLOT_LAYOUT,
        coloraxis_showscale=False,
        xaxis=dict(**GRID),
        yaxis=dict(autorange="reversed", color="#5b21b6"),
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
        textfont_color="#2d1b69",
        textfont_size=11,
        marker_line_color="#ffffff",
        marker_line_width=1,
    )
    fig_bot.update_layout(
        **PLOT_LAYOUT,
        coloraxis_showscale=False,
        xaxis=dict(**GRID),
        yaxis=dict(autorange="reversed", color="#5b21b6"),
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
