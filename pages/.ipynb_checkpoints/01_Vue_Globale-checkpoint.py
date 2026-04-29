"""
╔══════════════════════════════════════════════════════════════════╗
║   PLATEFORME INFLUENCEURS TUNISIE — Vue Globale                 ║
║   Dashboard | Classifier | Recommandation | Classement          ║
║   Thème : Salmverse                                             ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# ─── Chemins absolus ─────────────────────────────────────────────────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.dirname(_HERE)
_DATA   = os.path.join(_ROOT, "data")
_MODELS = os.path.join(_ROOT, "models")

def _data(f):  return os.path.join(_DATA,   f)
def _model(f): return os.path.join(_MODELS, f)

# ── Configuration page ──
st.set_page_config(
    page_title="Dashboard & Classifier — Influenceurs TN",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ── Fond sombre + styles globaux ──
st.markdown("""
<style>
    /* Fond général sombre */
    .stApp {
        background: linear-gradient(160deg, #1a0a24 0%, #250e2c 25%, #2d1040 50%, #1a0a24 100%) !important;
        background-attachment: fixed;
    }
    
    /* Forcer le fond des conteneurs */
    .main .block-container {
        background: transparent !important;
    }
    
    /* Texte clair par défaut */
    p, span, div, label {
        color: #e0d5f5;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  CSS SALMVERSE
# ══════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Sora:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #e0d5f5;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Sora', sans-serif;
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #3d1a4a 0%, #2d1040 50%, #1a0a24 100%) !important;
        border-right: 1px solid rgba(246,165,192,0.3) !important;
    }
    [data-testid="stSidebar"] * {
        color: #e0d5f5 !important;
    }

    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.1) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        border-radius: 14px !important;
        padding: 18px 20px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.25) !important;
        transition: all 0.25s !important;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(246,165,192,0.4) !important;
        box-shadow: 0 8px 25px rgba(246,165,192,0.2) !important;
        transform: translateY(-2px);
    }
    [data-testid="stMetricLabel"] {
        color: #f6a5c0 !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-family: 'Sora', sans-serif !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #f6a5c0, #cc8db3) !important;
        color: #250e2c !important;
        border: none !important;
        border-radius: 30px !important;
        font-weight: 700 !important;
        padding: 10px 24px !important;
        font-size: 14px !important;
        box-shadow: 0 5px 18px rgba(246,165,192,0.35) !important;
        transition: all 0.3s !important;
    }
    .stButton > button:hover {
        box-shadow: 0 8px 25px rgba(246,165,192,0.5) !important;
        transform: translateY(-2px) !important;
    }
    .stDownloadButton button {
        background: linear-gradient(135deg, #837ab6, #9d85b6) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 30px !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 14px rgba(131,122,182,0.3) !important;
        transition: all 0.3s !important;
    }
    .stDownloadButton button:hover {
        box-shadow: 0 6px 20px rgba(131,122,182,0.5) !important;
        transform: translateY(-2px) !important;
    }

    .stNumberInput > div > div > input,
    .stSelectbox > div > div,
    [data-testid="stSelectbox"] > div,
    .stTextInput > div > div > input,
    .stSlider > div {
        background: rgba(255,255,255,0.1) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
    }

    [data-testid="stDataFrame"] {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 12px !important;
        overflow: hidden;
    }
    [data-testid="stDataFrame"] th {
        background: rgba(246,165,192,0.15) !important;
        color: #f6a5c0 !important;
        font-weight: 600 !important;
        font-size: 12px !important;
    }
    [data-testid="stDataFrame"] td {
        color: #e0d5f5 !important;
        font-size: 13px !important;
    }

    .influencer-card {
        background: rgba(255,255,255,0.08) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        margin-bottom: 12px !important;
        transition: all 0.25s !important;
    }
    .influencer-card:hover {
        background: rgba(255,255,255,0.14) !important;
        border-color: rgba(246,165,192,0.4) !important;
        box-shadow: 0 6px 20px rgba(246,165,192,0.15) !important;
    }

    .badge-mega  { background: rgba(245,158,11,0.2); color: #fbbf24; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; border:1px solid rgba(245,158,11,0.3); display:inline-block; }
    .badge-macro { background: rgba(59,130,246,0.2); color: #60a5fa; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; border:1px solid rgba(59,130,246,0.3); display:inline-block; }
    .badge-micro { background: rgba(16,185,129,0.2); color: #34d399; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; border:1px solid rgba(16,185,129,0.3); display:inline-block; }
    .badge-nano  { background: rgba(107,114,128,0.2); color: #9ca3af; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; border:1px solid rgba(107,114,128,0.3); display:inline-block; }

    .warning-box {
        background: rgba(127,29,29,0.6);
        border: 1px solid #ef4444;
        border-radius: 8px;
        padding: 12px 16px;
        color: #fca5a5 !important;
        margin-top: 12px;
        backdrop-filter: blur(8px);
    }

    .page-title {
        font-family: 'Sora', sans-serif;
        font-size: 26px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 4px;
    }
    .page-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 13px;
        color: #c4b5fd;
        margin-bottom: 20px;
    }

    hr {
        border-color: rgba(246,165,192,0.25) !important;
    }

    [data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #f6a5c0, #cc8db3) !important;
        color: #250e2c !important;
        border: none !important;
        border-radius: 30px !important;
        font-weight: 700 !important;
        padding: 12px 24px !important;
        font-size: 15px !important;
        box-shadow: 0 5px 18px rgba(246,165,192,0.35) !important;
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  CHARGEMENT DES DONNEES ET MODELES
# ══════════════════════════════════════════════════════════════════

@st.cache_data
def charger_donnees():
    for chemin in [_data("dataset_ml.csv"), _data("dataset_clean.csv"),
                   _data("final_dataset.csv"), os.path.join(_ROOT,"dataset_ml.csv")]:
        if os.path.exists(chemin):
            df = pd.read_csv(chemin)
            cols_num = ["instagram_followers", "tiktok_followers",
                        "youtube_subscribers", "score_influence",
                        "audience_totale", "instagram_posts",
                        "instagram_following"]
            for c in cols_num:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
            if "audience_totale" not in df.columns:
                df["audience_totale"] = (
                    df.get("instagram_followers", 0) +
                    df.get("tiktok_followers", 0) +
                    df.get("youtube_subscribers", 0)
                )
            return df
    return pd.DataFrame()


@st.cache_resource
def charger_modeles():
    try:
        return {
            "modele":   joblib.load(_model("meilleur_modele.pkl")),
            "scaler":   joblib.load(_model("scaler.pkl")),
            "encoder":  joblib.load(_model("label_encoder.pkl")),
            "features": joblib.load(_model("features.pkl")),
        }
    except Exception:
        return None


df = charger_donnees()
modeles = charger_modeles()

COULEURS_NIVEAU = {
    "Mega":  "#f59e0b",
    "Macro": "#3b82f6",
    "Micro": "#10b981",
    "Nano":  "#6b7280",
}

SECTEURS = {
    "Tous secteurs":    [],
    "Mode / Beaute":    ["Fashion", "Modeling", "Makeup Artist"],
    "Lifestyle":        ["Lifestyle"],
    "Food / Cuisine":   ["Food"],
    "Sport / Fitness":  ["Sport"],
    "Musique":          ["Singer", "Rap"],
    "Humour":           ["Humor", "Video Blogger"],
    "Cinema / TV":      ["Actors", "TV Host"],
    "Voyage":           ["Travel"],
    "Sante":            ["Doctor"],
    "Photo / Art":      ["Photographer"],
}


# ══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<p style="font-family:\'Sora\',sans-serif;font-size:16px;font-weight:700;color:#ffffff;">🌍 Influenceurs TN</p>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio("Navigation", [
        "Dashboard",
        "Classifier un influenceur",
        "Recommandation marques",
        "Classement complet",
    ], label_visibility="collapsed")

    st.markdown("---")
    if not df.empty:
        st.markdown(f"**{len(df):,}** influenceurs")
        st.markdown(f"**{df['niveau'].nunique()}** niveaux")
        st.markdown(f"**{df['categorie'].nunique()}** categories")
    st.markdown("---")
    st.caption("Plateforme IA — Tunisie 2026")


# ══════════════════════════════════════════════════════════════════
#  PAGE 1 : DASHBOARD
# ══════════════════════════════════════════════════════════════════

if page == "Dashboard":
    st.markdown('<p class="page-title">📊 Tableau de bord</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Vue d\'ensemble des influenceurs tunisiens</p>', unsafe_allow_html=True)

    if df.empty:
        st.error("Dataset introuvable")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total influenceurs", f"{len(df):,}")
    c2.metric("Sur Instagram", f"{(df['instagram_followers']>0).sum():,}")
    c3.metric("Sur TikTok", f"{(df['tiktok_followers']>0).sum():,}")
    c4.metric("Sur YouTube", f"{(df['youtube_subscribers']>0).sum():,}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<h3 style="color:#ffffff;">📈 Distribution par niveau</h3>', unsafe_allow_html=True)
        niv = df["niveau"].value_counts()
        fig = go.Figure(go.Bar(
            x=niv.index, y=niv.values,
            marker_color=[COULEURS_NIVEAU.get(n, "#6b7280") for n in niv.index],
            text=niv.values, textposition="outside",
            textfont=dict(color="#ffffff", size=12),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c4b5fd", height=320, margin=dict(l=10, r=10, t=30, b=10),
            showlegend=False,
            xaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#e0d5f5")),
            yaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#e0d5f5")),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<h3 style="color:#ffffff;">🏷️ Top categories</h3>', unsafe_allow_html=True)
        cats = df["categorie"].value_counts().head(8)
        fig2 = go.Figure(go.Bar(
            x=cats.values, y=cats.index, orientation="h",
            marker=dict(color=cats.values, colorscale=[[0, '#837ab6'], [1, '#f6a5c0']]),
            text=cats.values, textposition="outside",
            textfont=dict(color="#ffffff", size=11),
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c4b5fd", height=320, margin=dict(l=10, r=40, t=30, b=10),
            xaxis=dict(gridcolor="rgba(255,255,255,0.08)", showticklabels=False),
            yaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#e0d5f5")),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<h3 style="color:#ffffff;">📊 Instagram vs TikTok</h3>', unsafe_allow_html=True)
    df_plot = df[(df["instagram_followers"] > 0)].head(300).copy()
    df_plot["score_affiche"] = df_plot["score_influence"].clip(5, 100)

    fig3 = px.scatter(
        df_plot, x="instagram_followers", y="tiktok_followers",
        size="score_affiche", color="niveau", color_discrete_map=COULEURS_NIVEAU,
        hover_data=["nom", "username", "categorie"] if "nom" in df_plot.columns else ["username"],
        labels={"instagram_followers": "IG", "tiktok_followers": "TK", "niveau": "Niveau"},
        size_max=30,
    )
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#c4b5fd", height=420,
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#e0d5f5")),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#e0d5f5")),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e0d5f5")),
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<h3 style="color:#ffffff;">🏆 Top 5 influenceurs</h3>', unsafe_allow_html=True)
    top5 = df.sort_values("score_influence", ascending=False).head(5)
    for _, row in top5.iterrows():
        niv = row.get("niveau", "")
        badge_cls = f"badge-{niv.lower()}" if niv.lower() in ["mega", "macro", "micro", "nano"] else "badge-nano"
        st.markdown(f"""
        <div class="influencer-card">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <div>
                    <span style="font-size:15px;font-weight:600;color:#ffffff">@{row["username"]}</span>
                    <span style="color:#c4b5fd;font-size:12px;margin-left:8px">{row.get("nom","")}</span>
                </div>
                <span class="{badge_cls}">{niv}</span>
            </div>
            <div style="margin-top:8px;display:flex;gap:20px;flex-wrap:wrap">
                <span style="color:#c4b5fd;font-size:12px">📸 <b style="color:#ffffff">{int(row.get("instagram_followers",0)):,}</b></span>
                <span style="color:#c4b5fd;font-size:12px">🎵 <b style="color:#ffffff">{int(row.get("tiktok_followers",0)):,}</b></span>
                <span style="color:#c4b5fd;font-size:12px">⭐ <b style="color:#f6a5c0">{int(row.get("score_influence",0))}/100</b></span>
                <span style="color:#c4b5fd;font-size:12px">🏷️ {row.get("categorie","")}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  PAGE 2 : CLASSIFIER
# ══════════════════════════════════════════════════════════════════

elif page == "Classifier un influenceur":
    st.markdown('<p class="page-title">🤖 Classifier un influenceur</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Entrez les statistiques pour predire le niveau avec le modele ML</p>', unsafe_allow_html=True)

    if modeles is None:
        st.error("Modele ML introuvable.")
        st.stop()

    with st.form("form_prediction"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p style="color:#f6a5c0;font-weight:600;">📸 Instagram</p>', unsafe_allow_html=True)
            ig_f = st.number_input("Followers", 0, 50_000_000, 100_000, 10_000, key="ig_f")
            ig_g = st.number_input("Following", 0, 5_000_000, 1_000, 100, key="ig_g")
            ig_p = st.number_input("Posts", 0, 20_000, 200, 10, key="ig_p")
        with col2:
            st.markdown('<p style="color:#f6a5c0;font-weight:600;">🎵 TikTok & ▶️ YouTube</p>', unsafe_allow_html=True)
            tk_f = st.number_input("TikTok followers", 0, 50_000_000, 0, 10_000, key="tk_f")
            yt_s = st.number_input("YouTube abonnes", 0, 50_000_000, 0, 10_000, key="yt_s")

        CATS = {"Actors":0,"Doctor":1,"Fashion":2,"Food":3,"Humor":4,"Lifestyle":5,"Makeup Artist":6,"Modeling":7,"Photographer":8,"Rap":9,"Singer":10,"Sport":11,"TV Host":12,"Travel":13,"Video Blogger":14}
        cat_nom = st.selectbox("Categorie", list(CATS.keys()), index=5)
        cat_enc = CATS[cat_nom]
        submitted = st.form_submit_button("🔍 Analyser", use_container_width=True)

    if submitted:
        features = modeles["features"]
        row = {f: 0 for f in features}
        row.update({"instagram_followers":ig_f,"instagram_following":ig_g,"instagram_posts":ig_p,"tiktok_followers":tk_f,"youtube_subscribers":yt_s,"categorie_encoded":cat_enc})
        audience = ig_f + tk_f + yt_s
        ratio_ff = ig_g / (ig_f + 1)
        nb_plat = sum([ig_f>0, tk_f>0, yt_s>0])
        if "audience_totale" in features: row["audience_totale"] = audience
        if "ratio_ff" in features: row["ratio_ff"] = ratio_ff
        if "nb_plateformes" in features: row["nb_plateformes"] = nb_plat

        X = pd.DataFrame([row])[features].fillna(0)
        X_s = modeles["scaler"].transform(X)
        niveau_code = modeles["modele"].predict(X_s)[0]
        niveau = modeles["encoder"].inverse_transform([niveau_code])[0]
        probas = modeles["modele"].predict_proba(X_s)[0]
        confiance = round(max(probas)*100, 1)

        st.markdown("---")
        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Niveau predit", niveau)
        col_r2.metric("Confiance ML", f"{confiance}%")
        col_r3.metric("Audience totale", f"{audience:,}")

        df_prob = pd.DataFrame({"Niveau":modeles["encoder"].classes_,"Proba":[round(p*100,1) for p in probas]}).sort_values("Proba", ascending=True)
        fig_prob = go.Figure(go.Bar(x=df_prob["Proba"], y=df_prob["Niveau"], orientation="h",
            marker_color=[COULEURS_NIVEAU.get(n,"#6b7280") for n in df_prob["Niveau"]],
            text=[f"{p}%" for p in df_prob["Proba"]], textposition="outside",
            textfont=dict(color="#ffffff", size=12)))
        fig_prob.update_layout(title=dict(text="Probabilites par niveau", font=dict(color="#ffffff")),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c4b5fd", height=250, margin=dict(l=10,r=60,t=40,b=10),
            xaxis=dict(range=[0,110], gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.08)"))
        st.plotly_chart(fig_prob, use_container_width=True)

        conseils = {"Mega":("Mega-influenceur 🌟","Ideal grandes campagnes",">10 000 TND"),
                    "Macro":("Macro-influenceur 💎","Bon rapport qualite/prix","2 000-10 000 TND"),
                    "Micro":("Micro-influenceur 🎯","Excellent engagement","500-2 000 TND"),
                    "Nano":("Nano-influenceur 💫","Communaute tres engagee","100-500 TND")}
        titre_c, desc_c, budget_c = conseils.get(niveau, ("","",""))
        couleur_c = COULEURS_NIVEAU.get(niveau, "#6b7280")

        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.08);backdrop-filter:blur(12px);
                    border:1.5px solid {couleur_c};border-radius:14px;padding:20px;margin-top:16px;">
            <div style="font-size:18px;font-weight:700;color:{couleur_c};">{titre_c}</div>
            <div style="color:#c4b5fd;margin-top:6px;">{desc_c}</div>
            <div style="color:#ffffff;margin-top:8px;font-weight:600;">💰 {budget_c}</div>
        </div>
        """, unsafe_allow_html=True)

        if ratio_ff > 2.0:
            st.markdown(f'<div class="warning-box">⚠️ Compte suspect (ratio={ratio_ff:.1f})</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  PAGE 3 : RECOMMANDATION
# ══════════════════════════════════════════════════════════════════

elif page == "Recommandation marques":
    st.markdown('<p class="page-title">🎯 Recommandation pour marques</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Trouvez les meilleurs influenceurs pour votre campagne</p>', unsafe_allow_html=True)

    if df.empty:
        st.error("Dataset introuvable")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        secteur = st.selectbox("Secteur", list(SECTEURS.keys()))
    with col2:
        niveau_f = st.selectbox("Niveau", ["Tous niveaux","Mega","Macro","Micro","Nano"])
    with col3:
        top_n = st.slider("Nombre de resultats", 3, 20, 5)

    if st.button("🔍 Rechercher", use_container_width=True):
        df_rec = df.copy()
        cats = SECTEURS.get(secteur, [])
        if cats: df_rec = df_rec[df_rec["categorie"].isin(cats)]
        if niveau_f != "Tous niveaux": df_rec = df_rec[df_rec["niveau"] == niveau_f]
        df_rec = df_rec.sort_values("score_influence", ascending=False)

        st.markdown("---")
        st.markdown(f'<p style="color:#c4b5fd;"><strong style="color:#f6a5c0;">{len(df_rec)}</strong> influenceurs — {min(top_n,len(df_rec))} meilleurs</p>', unsafe_allow_html=True)

        if df_rec.empty:
            st.warning("Aucun resultat.")
        else:
            for i, (_, row) in enumerate(df_rec.head(top_n).iterrows(), 1):
                niv = row.get("niveau","")
                badge_c = f"badge-{niv.lower()}" if niv.lower() in ["mega","macro","micro","nano"] else "badge-nano"
                score = int(row.get("score_influence",0))
                ig_f_r = int(row.get("instagram_followers",0))
                tk_f_r = int(row.get("tiktok_followers",0))
                yt_s_r = int(row.get("youtube_subscribers",0))
                audience = int(row.get("audience_totale", ig_f_r+tk_f_r+yt_s_r))
                barre_w = score

                st.markdown(f"""
                <div class="influencer-card">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start">
                        <div>
                            <span style="color:#c4b5fd;font-size:12px">#{i}</span>
                            <span style="font-size:15px;font-weight:700;color:#ffffff;margin-left:8px">@{row["username"]}</span>
                            <span style="color:#c4b5fd;font-size:12px;margin-left:6px">— {row.get("nom","")}</span>
                        </div>
                        <div style="display:flex;gap:8px;align-items:center">
                            <span class="{badge_c}">{niv}</span>
                            <span style="color:#f6a5c0;font-weight:700;">{score}/100</span>
                        </div>
                    </div>
                    <div style="margin-top:10px;display:flex;gap:18px;flex-wrap:wrap">
                        <span style="color:#c4b5fd;font-size:12px">🏷️ <b style="color:#ffffff">{row.get("categorie","")}</b></span>
                        <span style="color:#c4b5fd;font-size:12px">📸 <b style="color:#ffffff">{ig_f_r:,}</b></span>
                        <span style="color:#c4b5fd;font-size:12px">🎵 <b style="color:#ffffff">{tk_f_r:,}</b></span>
                        <span style="color:#c4b5fd;font-size:12px">▶️ <b style="color:#ffffff">{yt_s_r:,}</b></span>
                        <span style="color:#c4b5fd;font-size:12px">👥 <b style="color:#ffffff">{audience:,}</b></span>
                    </div>
                    <div style="margin-top:10px;background:rgba(255,255,255,0.08);border-radius:6px;height:5px;">
                        <div style="width:{barre_w}%;height:5px;border-radius:6px;background:linear-gradient(90deg,#837ab6,#f6a5c0);"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            csv = df_rec.head(top_n).to_csv(index=False)
            st.download_button("📥 Telecharger CSV", csv, f"recommandations_{secteur.replace('/','-')}.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════
#  PAGE 4 : CLASSEMENT COMPLET
# ══════════════════════════════════════════════════════════════════

elif page == "Classement complet":
    st.markdown('<p class="page-title">🏆 Classement complet</p>', unsafe_allow_html=True)

    if df.empty:
        st.error("Dataset introuvable")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        cats_opts = ["Toutes"] + sorted(df["categorie"].dropna().unique().tolist())
        cat_f = st.selectbox("Categorie", cats_opts)
    with col2:
        niv_opts = ["Tous"] + sorted(df["niveau"].dropna().unique().tolist())
        niv_f = st.selectbox("Niveau", niv_opts)
    with col3:
        recherche = st.text_input("Rechercher", placeholder="@username ou nom...")

    df_f = df.copy()
    if cat_f != "Toutes":
        df_f = df_f[df_f["categorie"] == cat_f]
    if niv_f != "Tous":
        df_f = df_f[df_f["niveau"] == niv_f]
    if recherche:
        mask = df_f["username"].str.contains(recherche, case=False, na=False)
        if "nom" in df_f.columns:
            mask = mask | df_f["nom"].str.contains(recherche, case=False, na=False)
        df_f = df_f[mask]

    df_f = df_f.sort_values("score_influence", ascending=False)
    st.markdown(f'<p style="color:#c4b5fd;"><strong style="color:#f6a5c0;">{len(df_f)}</strong> influenceurs trouves</p>', unsafe_allow_html=True)

    cols_afficher = ["username", "nom", "categorie", "niveau",
                     "instagram_followers", "tiktok_followers",
                     "youtube_subscribers", "score_influence"]
    cols_afficher = [c for c in cols_afficher if c in df_f.columns]

    st.dataframe(
        df_f[cols_afficher].reset_index(drop=True),
        use_container_width=True,
        height=500,
        column_config={
            "instagram_followers": st.column_config.NumberColumn("IG Followers", format="%d"),
            "tiktok_followers": st.column_config.NumberColumn("TK Followers", format="%d"),
            "youtube_subscribers": st.column_config.NumberColumn("YT Abonnes", format="%d"),
            "score_influence": st.column_config.ProgressColumn("Score", min_value=0, max_value=100),
        }
    )

    csv = df_f[cols_afficher].to_csv(index=False)
    st.download_button("📥 Telecharger CSV", csv, "classement.csv", "text/csv")

# ── Footer ──
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:16px 0;">
    <p style="color:#9d85b6;font-size:11px;margin:0;">
        Plateforme IA — Tunisie 2026 · 
        <span style="color:#cc8db3;">Salmverse</span>
    </p>
</div>
""", unsafe_allow_html=True)