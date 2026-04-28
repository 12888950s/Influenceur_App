"""
╔══════════════════════════════════════════════════════════════════╗
║   PLATEFORME INFLUENCEURS TUNISIENS — Point d'entrée principal  ║
║   Thème : Salmverse                                             ║
║   Palette : #250e2c #837ab6 #9d85b6 #cc8db3 #f6a5c0 #f7c2ca   ║
║   Lancement : streamlit run main_app.py                         ║
╚══════════════════════════════════════════════════════════════════╝

Structure attendue :
  main_app.py
  assets/
    style.css
    logo_horizontal.svg
  pages/
    01_vue_globale.py
    02_sentiments.py
    03_classement.py
    04_classifier_ml.py
  data/
    user_sentiment_global_.csv
    dataset_ml.csv
  models/
    meilleur_modele.pkl
    scaler.pkl
    label_encoder.pkl
    features.pkl
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import base64
from auth import render_auth_page, is_logged_in, current_user, logout

# ─── Résolution des chemins absolus ──────────────────────────────────────────
_ROOT   = os.path.dirname(os.path.abspath(__file__))
_DATA   = os.path.join(_ROOT, "data")
_MODELS = os.path.join(_ROOT, "models")
_ASSETS = os.path.join(_ROOT, "assets")

def _data(f):  return os.path.join(_DATA,   f)
def _model(f): return os.path.join(_MODELS, f)
def _asset(f): return os.path.join(_ASSETS, f)

# ══════════════════════════════════════════════════════════════════
#  CONFIG PAGE
# ══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Influence TN — Classification",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
#  AUTHENTIFICATION — Bloque l'accès si non connecté
# ══════════════════════════════════════════════════════════════════

if not is_logged_in():
    render_auth_page()
    st.stop()

# ══════════════════════════════════════════════════════════════════
#  CSS GLOBAL — Chargement depuis fichier externe
# ══════════════════════════════════════════════════════════════════

def load_css(file_path):
    """Charge un fichier CSS et l'injecte dans l'application."""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ Fichier CSS introuvable : {file_path}")

# Charger le CSS externe
css_path = _asset("style.css")
load_css(css_path)

# ══════════════════════════════════════════════════════════════════
#  LOGO EN HAUT DE LA SIDEBAR (fixe, au-dessus de la navigation)
# ══════════════════════════════════════════════════════════════════




# ══════════════════════════════════════════════════════════════════
#  CSS POUR LE LOGO DANS LA SIDEBAR
# ══════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* Container du logo dans la sidebar */
    [data-testid="stSidebar"] .sidebar-logo-wrapper {
        background: linear-gradient(160deg, rgba(255,255,255,0.18) 0%, rgba(246,165,192,0.12) 100%);
        border: 1.5px solid rgba(246,165,192,0.35);
        border-radius: 14px;
        padding: 14px 12px;
        margin-bottom: 14px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  CHARGEMENT & FUSION DES DONNÉES (mis en cache)
# ══════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Chargement des données…")
def load_sentiment_data() -> pd.DataFrame:
    """Charge user_sentiment_global_.csv depuis data/ ou le dossier courant."""
    for path in [_data("user_sentiment_global_.csv"), os.path.join(_ROOT, "user_sentiment_global_.csv")]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["user_name"] = df["user_name"].str.strip().str.lower()

            # Plateforme
            if "youtube_total_comments" not in df.columns:
                df["youtube_total_comments"] = 0
            if "tiktok_total_comments" not in df.columns:
                df["tiktok_total_comments"] = 0
            df["platform"] = df.apply(
                lambda r: "YouTube + TikTok"
                if r["youtube_total_comments"] > 0 and r["tiktok_total_comments"] > 0
                else ("YouTube" if r["youtube_total_comments"] > 0 else "TikTok"),
                axis=1,
            )

            # Catégorie sentiment
            df["sentiment_category"] = pd.cut(
                df["global_score"],
                bins=[-100, 0, 40, 60, 80, 101],
                labels=["Négatif 🔴", "Faible 🟠", "Moyen 🟡", "Bon 🟢", "Excellent ⭐"],
            )

            # Pourcentages dominants (YouTube prioritaire)
            for col in ["pos_pct", "neg_pct", "neu_pct"]:
                prefix = col.replace("_pct", "")
                yt_col   = f"youtube_{prefix}_pct"
                tt_col   = f"tiktok_{prefix}_pct"
                if yt_col not in df.columns:
                    df[yt_col] = float("nan")
                if tt_col not in df.columns:
                    df[tt_col] = float("nan")
                df[col] = df.apply(
                    lambda r, yc=yt_col, tc=tt_col: r[yc]
                    if r.get("youtube_total_comments", 0) > 0
                    else r[tc],
                    axis=1,
                )

            df["total_comments"] = (
                df["youtube_total_comments"].fillna(0)
                + df["tiktok_total_comments"].fillna(0)
            )
            return df
    return pd.DataFrame()


@st.cache_data(show_spinner="Chargement du ranking…")
def load_influence_data() -> pd.DataFrame:
    """Charge le dataset ML (ranking, reach, plateformes sociales)."""
    for path in [
        _data("dataset_ml.csv"),
        _data("dataset_clean.csv"),
        _data("final_dataset.csv"),
        os.path.join(_ROOT, "dataset_ml.csv"),
    ]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            cols_num = [
                "instagram_followers", "tiktok_followers",
                "youtube_subscribers", "score_influence",
                "audience_totale", "instagram_posts", "instagram_following",
            ]
            for c in cols_num:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

            if "audience_totale" not in df.columns:
                df["audience_totale"] = (
                    df.get("instagram_followers", 0)
                    + df.get("tiktok_followers", 0)
                    + df.get("youtube_subscribers", 0)
                )

            if "username" in df.columns:
                df["username_key"] = df["username"].str.strip().str.lower()

            return df
    return pd.DataFrame()


@st.cache_data(show_spinner="Fusion des datasets…")
def load_merged_data() -> pd.DataFrame:
    """Joint les deux datasets sur user_name ↔ username."""
    df_sent = load_sentiment_data()
    df_inf  = load_influence_data()

    if df_sent.empty or df_inf.empty:
        return df_sent if not df_sent.empty else df_inf

    sent_cols = {
        c: f"sent_{c}"
        for c in df_sent.columns
        if c not in ("user_name",)
    }
    df_sent_r = df_sent.rename(columns=sent_cols)

    df_merged = df_inf.merge(
        df_sent_r,
        left_on="username_key",
        right_on="user_name",
        how="left",
    )

    df_merged["has_sentiment"] = df_merged["sent_global_score"].notna()
    return df_merged


@st.cache_resource(show_spinner="Chargement des modèles ML…")
def load_models() -> dict | None:
    """Charge les modèles joblib depuis models/."""
    try:
        return {
            "modele":   joblib.load(_model("meilleur_modele.pkl")),
            "scaler":   joblib.load(_model("scaler.pkl")),
            "encoder":  joblib.load(_model("label_encoder.pkl")),
            "features": joblib.load(_model("features.pkl")),
        }
    except Exception:
        return None


# ── Constantes partagées ──────────────────────────────────────────

COULEURS_NIVEAU = {
    "Mega":  "#f59e0b",
    "Macro": "#3b82f6",
    "Micro": "#10b981",
    "Nano":  "#6b7280",
}

SECTEURS = {
    "Tous secteurs":   [],
    "Mode / Beauté":   ["Fashion", "Modeling", "Makeup Artist"],
    "Lifestyle":       ["Lifestyle"],
    "Food / Cuisine":  ["Food"],
    "Sport / Fitness": ["Sport"],
    "Musique":         ["Singer", "Rap"],
    "Humour":          ["Humor", "Video Blogger"],
    "Cinéma / TV":     ["Actors", "TV Host"],
    "Voyage":          ["Travel"],
    "Santé":           ["Doctor"],
    "Photo / Art":     ["Photographer"],
}

# ══════════════════════════════════════════════════════════════════
#  MISE EN SESSION
# ══════════════════════════════════════════════════════════════════

if "df_sentiment" not in st.session_state:
    st.session_state["df_sentiment"] = load_sentiment_data()

if "df_influence" not in st.session_state:
    st.session_state["df_influence"] = load_influence_data()

if "df_merged" not in st.session_state:
    st.session_state["df_merged"] = load_merged_data()

if "models" not in st.session_state:
    st.session_state["models"] = load_models()


# ══════════════════════════════════════════════════════════════════
#  SIDEBAR PRINCIPALE
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    # ── LOGO (tout en haut de la sidebar) ──
    logo_svg_path = os.path.join(_ROOT, "assets", "logo_horizontal.svg")
    svg_found = False
    
    for path in [
        logo_svg_path,
        os.path.join(_ROOT, "assets", "logo_horizontal.svg"),
        os.path.join(_ROOT, "logo_horizontal.svg"),
    ]:
        if os.path.exists(path):
            svg_found = True
            logo_final_path = path
            break
    
    if svg_found:
        with open(logo_final_path, "r", encoding="utf-8") as f:
            svg_content = f.read()
        st.markdown(f"""
        <div class="sidebar-logo-wrapper">
            <div style="width:100%;max-width:180px;margin:0 auto;">
                {svg_content}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="sidebar-logo-wrapper">
            <div style="font-size:22px;color:#f6a5c0">✨</div>
            <div style="font-family:'Sora',sans-serif;font-size:14px;font-weight:700;color:#ffffff;margin-top:2px">
                Influence TN
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ── Utilisateur connecté ──
    _uname = current_user()
    # ... (le reste de votre code sidebar)
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;padding:6px 2px;margin-bottom:4px">'
        f'<div class="user-avatar">{_uname[0].upper()}</div>'
        f'<div>'
        f'<div style="font-size:13px;font-weight:700;color:#ffffff">{_uname}</div>'
        f'<div style="font-size:10px;color:#cc8db3">🟢 Connecté</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )
    if st.button("🚪 Se déconnecter", use_container_width=True, key="btn_logout"):
        logout()

    st.markdown("---")

    # ── Statut des données ──
    df_s = st.session_state["df_sentiment"]
    df_i = st.session_state["df_influence"]
    df_m = st.session_state["df_merged"]
    models = st.session_state["models"]

    st.markdown("**📊 État des données**")

    def _status(ok, label):
        color = "#4ade80" if ok else "#f87171"
        icon  = "●" if ok else "○"
        st.markdown(
            f'<div style="font-size:12px;color:{color};margin:4px 0">'
            f'{icon} {label}</div>',
            unsafe_allow_html=True,
        )

    _status(not df_s.empty, f"Sentiments — {len(df_s):,} influenceurs")
    _status(not df_i.empty, f"Ranking — {len(df_i):,} influenceurs")
    _status(models is not None, "Modèle ML chargé")

    if not df_m.empty and "has_sentiment" in df_m.columns:
        pct = int(df_m["has_sentiment"].mean() * 100)
        st.markdown(
            f'<div style="font-size:12px;color:#e0d5f5;margin-top:6px">'
            f'🔗 Fusion : {pct}% matchés</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Filtres globaux ──
    st.markdown("**🔍 Filtres globaux**")

    if not df_i.empty and "niveau" in df_i.columns:
        niveaux_dispo = sorted(df_i["niveau"].dropna().unique().tolist())
        filtre_niveau = st.multiselect(
            "Niveau d'influence",
            options=niveaux_dispo,
            default=niveaux_dispo,
            key="filtre_niveau",
        )
    else:
        filtre_niveau = []

    if not df_i.empty and "categorie" in df_i.columns:
        cats_dispo = sorted(df_i["categorie"].dropna().unique().tolist())
        filtre_categorie = st.multiselect(
            "Catégorie",
            options=cats_dispo,
            default=cats_dispo,
            key="filtre_categorie",
        )
    else:
        filtre_categorie = []

    filtre_recherche = st.text_input(
        "🔎 Recherche rapide",
        placeholder="Nom ou @username…",
        key="filtre_recherche",
    )

    st.markdown("---")

    # ── Footer sidebar ──
    st.markdown(
        '<div class="sidebar-footer">'
        '📡 Données : Instagram, TikTok & YouTube<br>'
        '🧠 Modèle ML : Random Forest<br>'
        '💬 NLP : Arabe & Français<br>'
        '<span style="color:#cc8db3">© 2025 Salmverse</span>'
        '</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════
#  PAGE D'ACCUEIL
# ══════════════════════════════════════════════════════════════════

# ── Hero Banner Salmverse ──
n_inf = len(df_i) if not df_i.empty else len(df_s)
st.markdown(f"""
<div class="hero">
  <h1>✨ Classification des Influenceurs Tunisiens</h1>
  <p>
    Analyse intelligente · Sentiments des followers · Ranking &amp; Reach · ML Classification ·
    <strong style="color:#ffffff">{n_inf:,} influenceurs indexés</strong>
  </p>
  <div style="margin-top:16px;display:flex;gap:10px;flex-wrap:wrap">
    <span class="hero-tag">📊 Vue Globale</span>
    <span class="hero-tag">💬 Analyse Sentiments</span>
    <span class="hero-tag">🏆 Classement & Reach</span>
    <span class="hero-tag">🤖 Classifier ML</span>
    <span class="hero-tag">🎯 Recommandation</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPIs en Glassmorphism ──
c1, c2, c3, c4, c5 = st.columns(5)
if not df_i.empty:
    c1.metric("👥 Total", f"{len(df_i):,}")
    c2.metric("📸 Instagram", f"{(df_i.get('instagram_followers', pd.Series(dtype=float)) > 0).sum():,}")
    c3.metric("🎵 TikTok", f"{(df_i.get('tiktok_followers', pd.Series(dtype=float)) > 0).sum():,}")
    c4.metric("▶️ YouTube", f"{(df_i.get('youtube_subscribers', pd.Series(dtype=float)) > 0).sum():,}")
elif not df_s.empty:
    c1.metric("👥 Influenceurs", f"{len(df_s):,}")
    c2.metric("📈 Score moyen", f"{df_s['global_score'].mean():.1f}")
    c3.metric("🏆 Score max", f"{df_s['global_score'].max():.1f}")
    c4.metric("💚 % Positif moy.", f"{df_s['pos_pct'].mean():.1f}%")

if not df_s.empty and "pos_pct" in df_s.columns:
    val = df_s["pos_pct"].mean()
    c5.metric("💚 % Positif", f"{val:.1f}%" if pd.notna(val) else "—")
elif not df_i.empty and "score_influence" in df_i.columns:
    c5.metric("⭐ Score infl.", f"{df_i['score_influence'].mean():.1f}")

st.markdown("<br>", unsafe_allow_html=True)

# ── Section À propos ───────────────────────────────────────────
with st.container():
    st.markdown("""
    <div class="about-box">
      <div style="font-family:'Sora',sans-serif;font-size:18px;font-weight:700;
                  color:#ffffff;margin-bottom:12px">
        📌 À propos de cette plateforme
      </div>
      <p style="color:#e0d5f5;font-size:14px;line-height:1.8;margin:0 0 14px 0">
        Cette plateforme analyse <strong style="color:#ffffff">les influenceurs tunisiens</strong> actifs sur
        <strong style="color:#ffffff">Instagram</strong>, <strong style="color:#ffffff">TikTok</strong> et <strong style="color:#ffffff">YouTube</strong>.
        Elle combine l'analyse de sentiment NLP (arabe &amp; français) des commentaires
        avec un modèle de Machine Learning (Random Forest) pour classifier les influenceurs
        par niveau d'audience et de reach.
      </p>
      <div style="display:flex;gap:24px;flex-wrap:wrap;margin-top:14px">
        <div style="display:flex;align-items:center;gap:8px">
          <div style="width:12px;height:12px;border-radius:50%;background:#f59e0b"></div>
          <span style="font-size:13px;color:#e0d5f5"><strong style="color:#ffffff">Mega</strong> — &gt; 500K</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px">
          <div style="width:12px;height:12px;border-radius:50%;background:#3b82f6"></div>
          <span style="font-size:13px;color:#e0d5f5"><strong style="color:#ffffff">Macro</strong> — 100K–500K</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px">
          <div style="width:12px;height:12px;border-radius:50%;background:#10b981"></div>
          <span style="font-size:13px;color:#e0d5f5"><strong style="color:#ffffff">Micro</strong> — 10K–100K</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px">
          <div style="width:12px;height:12px;border-radius:50%;background:#6b7280"></div>
          <span style="font-size:13px;color:#e0d5f5"><strong style="color:#ffffff">Nano</strong> — &lt; 10K</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Cartes de navigation interactives ──────────────────────────
st.markdown("""
<div style="font-family:'Sora',sans-serif;font-size:13px;font-weight:700;
            text-transform:uppercase;letter-spacing:.08em;color:#f6a5c0;
            margin-bottom:14px">
  📂 Choisissez une section
</div>
""", unsafe_allow_html=True)

nav1, nav2 = st.columns(2)
nav3, nav4 = st.columns(2)

CARDS = [
    (nav1, "📊", "Vue Globale & Sentiments",
     "Analyse des sentiments (positif / négatif / neutre) par influenceur · YouTube & TikTok",
     "#837ab6"),
    (nav2, "🏆", "Dashboard & Classement",
     "KPIs, distribution par niveau et catégorie, scatter Instagram vs TikTok, Top 5",
     "#cc8db3"),
    (nav3, "🤖", "Classifier un Influenceur",
     "Entrez les stats d'un influenceur — le modèle ML prédit son niveau avec un score de confiance",
     "#f6a5c0"),
    (nav4, "🎯", "Recommandation Marques",
     "Trouvez les meilleurs influenceurs pour votre campagne par secteur et niveau",
     "#9d85b6"),
]

for col, icon, title, desc, color in CARDS:
    col.markdown(f"""
    <div class="nav-card">
      <div class="nav-card-accent" style="background:{color}"></div>
      <div class="nav-icon">{icon}</div>
      <div class="nav-title">{title}</div>
      <div class="nav-desc">{desc}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Guide d'utilisation ────────────────────────────────────────
with st.expander("ℹ️ Comment utiliser cette plateforme ?", expanded=False):
    col_guide1, col_guide2 = st.columns(2)
    with col_guide1:
        st.markdown("""
        ### 🧭 Navigation
        - **Vue Globale** → analyse des sentiments des commentaires YouTube & TikTok
        - **Sentiment** → dashboard général, classement complet
        - **Classifier ML** → prédiction du niveau d'influence
        - **Recommandation** → matching influenceurs/marques

        ### 🔍 Filtres (sidebar)
        - Filtrez par **niveau** (Mega / Macro / Micro / Nano)
        - Filtrez par **catégorie** (Fashion, Food, Sport...)
        - **Recherche** par nom ou @username
        """)
    with col_guide2:
        st.markdown("""
        ### 🤖 Classifier ML
        1. Renseignez les stats Instagram / TikTok / YouTube
        2. Le modèle **Random Forest** prédit le niveau
        3. Score de confiance affiché

        ### ⚠️ Alerte compte suspect
        Un ratio following/followers élevé déclenche une alerte.

        ### 📡 Sources
        Données collectées sur Instagram, TikTok & YouTube.
        NLP en arabe et français.
        """)

st.markdown("<br>", unsafe_allow_html=True)

# ── Aperçu fusion datasets ─────────────────────────────────────
if not df_m.empty and "has_sentiment" in df_m.columns:
    with st.expander("🔗 Aperçu de la fusion des datasets", expanded=False):
        n_match = int(df_m["has_sentiment"].sum())
        pct_match = int(df_m["has_sentiment"].mean() * 100)
        st.markdown(
            f"**{n_match:,}** influenceurs ont à la fois des données de ranking "
            f"**et** de sentiment ({pct_match}% des {len(df_m):,} au total)."
        )
        preview_cols = [c for c in [
            "username", "nom", "niveau", "score_influence",
            "sent_global_score", "sent_pos_pct", "sent_neg_pct", "has_sentiment",
        ] if c in df_m.columns]
        st.dataframe(df_m[preview_cols].head(10), use_container_width=True, height=260)

# ── Footer ─────────────────────────────────────────────────────
st.divider()
footer_col1, footer_col2, footer_col3 = st.columns([2, 1, 1])
with footer_col1:
    st.markdown(
        '<p style="color:#c4b5fd;font-size:12px;margin:0">'
        '© 2025 <strong style="color:#ffffff">Salmverse</strong> — Plateforme de Classification des Influenceurs Tunisiens</p>',
        unsafe_allow_html=True
    )
with footer_col2:
    st.markdown(
        '<p style="color:#9d85b6;font-size:12px;text-align:center;margin:0">'
        '📊 v1.0.0</p>',
        unsafe_allow_html=True
    )
with footer_col3:
    st.markdown(
        '<p style="color:#cc8db3;font-size:12px;text-align:right;margin:0">'
        '✨ Made with Streamlit</p>',
        unsafe_allow_html=True
    )