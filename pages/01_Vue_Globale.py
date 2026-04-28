"""
=======================================================
  PLATEFORME INFLUENCEURS TUNISIE — Interface Web
  Lancement : streamlit run dashboard_web.py
=======================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# ─── Chemins absolus (fonctionne peu importe le CWD de Streamlit) ─────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))   # .../project/pages
_ROOT   = os.path.dirname(_HERE)                        # .../project
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

# ── CSS personnalise ──
st.markdown("""
<style>
    /* Police et fond */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] .stRadio > label {
        color: #94a3b8 !important;
        font-size: 13px;
    }

    /* Cards metriques */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
    }
    [data-testid="metric-container"] label {
        color: #64748b !important;
        font-size: 12px !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }

    /* Titres */
    h1 { color: #f1f5f9 !important; font-weight: 700 !important; }
    h2 { color: #cbd5e1 !important; font-weight: 600 !important; }
    h3 { color: #94a3b8 !important; font-weight: 500 !important; }

    /* Boutons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        letter-spacing: 0.3px;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.9; }

    /* Inputs */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div,
    .stSlider > div {
        border-radius: 8px !important;
    }

    /* Badges niveaux */
    .badge-mega  { background:#fef3c7;color:#92400e;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600; }
    .badge-macro { background:#dbeafe;color:#1e40af;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600; }
    .badge-micro { background:#d1fae5;color:#065f46;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600; }
    .badge-nano  { background:#f3f4f6;color:#374151;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600; }

    /* Cards influenceurs */
    .influencer-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        transition: border-color 0.2s;
    }
    .influencer-card:hover { border-color: #6366f1; }

    /* Fond principal */
    .main { background-color: #0f172a !important; }
    .block-container { padding-top: 2rem !important; }

    /* Separateur */
    hr { border-color: #1e293b !important; }

    /* Resultat prediction */
    .prediction-result {
        background: #1e293b;
        border: 1px solid #6366f1;
        border-radius: 16px;
        padding: 24px;
        margin-top: 20px;
        text-align: center;
    }
    .prediction-niveau {
        font-size: 48px;
        font-weight: 700;
        color: #a5b4fc;
        margin: 8px 0;
    }
    .prediction-score {
        font-size: 20px;
        color: #64748b;
    }
    .warning-box {
        background: #7f1d1d;
        border: 1px solid #ef4444;
        border-radius: 8px;
        padding: 12px 16px;
        color: #fca5a5 !important;
        margin-top: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  CHARGEMENT DES DONNEES ET MODELES
# ══════════════════════════════════════════════════════

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
    except Exception as e:
        return None


df = charger_donnees()
modeles = charger_modeles()

COULEURS_NIVEAU = {
    "Mega":  "#f59e0b",
    "Macro": "#3b82f6",
    "Micro": "#10b981",
    "Nano":  "#6b7280",
}


# ══════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🌍 Influenceurs TN")
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


# ══════════════════════════════════════════════════════
#  PAGE 1 : DASHBOARD
# ══════════════════════════════════════════════════════

if page == "Dashboard":
    st.title("Tableau de bord")
    st.markdown("Vue d'ensemble des influenceurs tunisiens")

    if df.empty:
        st.error("Dataset introuvable")
        st.stop()

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total influenceurs",  f"{len(df):,}")
    c2.metric("Sur Instagram",
              f"{(df['instagram_followers']>0).sum():,}")
    c3.metric("Sur TikTok",
              f"{(df['tiktok_followers']>0).sum():,}")
    c4.metric("Sur YouTube",
              f"{(df['youtube_subscribers']>0).sum():,}")

    st.markdown("---")

    # Graphiques ligne 1
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution par niveau")
        niv = df["niveau"].value_counts()
        fig = go.Figure(go.Bar(
            x=niv.index,
            y=niv.values,
            marker_color=[COULEURS_NIVEAU.get(n, "#6b7280")
                          for n in niv.index],
            text=niv.values,
            textposition="outside",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8",
            height=300,
            margin=dict(l=0, r=0, t=20, b=0),
            showlegend=False,
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top categories")
        cats = df["categorie"].value_counts().head(8)
        fig2 = go.Figure(go.Bar(
            x=cats.values,
            y=cats.index,
            orientation="h",
            marker_color="#6366f1",
            text=cats.values,
            textposition="outside",
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8",
            height=300,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Scatter
    st.subheader("Instagram vs TikTok — taille = score d'influence")
    df_plot = df[(df["instagram_followers"] > 0)].head(300).copy()
    df_plot["score_affiche"] = df_plot["score_influence"].clip(5, 100)

    fig3 = px.scatter(
        df_plot,
        x="instagram_followers",
        y="tiktok_followers",
        size="score_affiche",
        color="niveau",
        color_discrete_map=COULEURS_NIVEAU,
        hover_data=["nom", "username", "categorie"] if "nom" in df_plot.columns else ["username"],
        labels={
            "instagram_followers": "Followers Instagram",
            "tiktok_followers":    "Followers TikTok",
            "niveau":              "Niveau",
        },
        size_max=30,
    )
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#94a3b8",
        height=400,
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Top 5
    st.subheader("Top 5 influenceurs")
    top5 = df.sort_values("score_influence", ascending=False).head(5)
    for _, row in top5.iterrows():
        niv = row.get("niveau", "")
        badge_cls = f"badge-{niv.lower()}"
        st.markdown(f"""
        <div class="influencer-card">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <div>
                    <span style="font-size:16px;font-weight:600;color:#e2e8f0">
                        @{row["username"]}
                    </span>
                    <span style="color:#64748b;font-size:13px;margin-left:8px">
                        {row.get("nom","")}
                    </span>
                </div>
                <span class="{badge_cls}">{niv}</span>
            </div>
            <div style="margin-top:8px;display:flex;gap:24px">
                <span style="color:#94a3b8;font-size:13px">
                    IG: <b style="color:#e2e8f0">{int(row.get("instagram_followers",0)):,}</b>
                </span>
                <span style="color:#94a3b8;font-size:13px">
                    TK: <b style="color:#e2e8f0">{int(row.get("tiktok_followers",0)):,}</b>
                </span>
                <span style="color:#94a3b8;font-size:13px">
                    Score: <b style="color:#a5b4fc">{int(row.get("score_influence",0))}/100</b>
                </span>
                <span style="color:#94a3b8;font-size:13px">
                    {row.get("categorie","")}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  PAGE 2 : CLASSIFIER UN INFLUENCEUR
# ══════════════════════════════════════════════════════

elif page == "Classifier un influenceur":
    st.title("Classifier un influenceur")
    st.markdown("Entrez les statistiques pour predire le niveau avec le modele ML")

    if modeles is None:
        st.error(f"❌ Modèle ML introuvable.\n\nChemin cherché : `{_model('meilleur_modele.pkl')}`\n\nVérifie que le dossier `models/` contient bien les fichiers `.pkl`.")
        st.stop()

    with st.form("form_prediction"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Instagram")
            ig_f = st.number_input("Followers",  0, 50_000_000, 100_000, 10_000,
                                    key="ig_f")
            ig_g = st.number_input("Following",  0, 5_000_000,  1_000,   100,
                                    key="ig_g")
            ig_p = st.number_input("Posts",      0, 20_000,     200,     10,
                                    key="ig_p")

        with col2:
            st.markdown("#### TikTok & YouTube")
            tk_f = st.number_input("TikTok followers",    0, 50_000_000, 0, 10_000,
                                    key="tk_f")
            yt_s = st.number_input("YouTube abonnes",     0, 50_000_000, 0, 10_000,
                                    key="yt_s")

        CATS = {
            "Actors": 0, "Doctor": 1, "Fashion": 2, "Food": 3,
            "Humor": 4, "Lifestyle": 5, "Makeup Artist": 6,
            "Modeling": 7, "Photographer": 8, "Rap": 9,
            "Singer": 10, "Sport": 11, "TV Host": 12,
            "Travel": 13, "Video Blogger": 14,
        }
        cat_nom = st.selectbox("Categorie", list(CATS.keys()), index=5)
        cat_enc = CATS[cat_nom]

        submitted = st.form_submit_button("Analyser", use_container_width=True)

    if submitted:
        features = modeles["features"]
        row = {f: 0 for f in features}
        row.update({
            "instagram_followers": ig_f,
            "instagram_following": ig_g,
            "instagram_posts":     ig_p,
            "tiktok_followers":    tk_f,
            "youtube_subscribers": yt_s,
            "categorie_encoded":   cat_enc,
        })

        # Features derivees
        audience  = ig_f + tk_f + yt_s
        ratio_ff  = ig_g / (ig_f + 1)
        nb_plat   = sum([ig_f > 0, tk_f > 0, yt_s > 0])

        if "audience_totale" in features:
            row["audience_totale"] = audience
        if "ratio_ff" in features:
            row["ratio_ff"] = ratio_ff
        if "nb_plateformes" in features:
            row["nb_plateformes"] = nb_plat

        X = pd.DataFrame([row])[features].fillna(0)
        X_s = modeles["scaler"].transform(X)

        niveau_code = modeles["modele"].predict(X_s)[0]
        niveau      = modeles["encoder"].inverse_transform([niveau_code])[0]
        probas      = modeles["modele"].predict_proba(X_s)[0]
        confiance   = round(max(probas) * 100, 1)

        # Affichage resultat
        st.markdown("---")

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Niveau predit",  niveau)
        col_r2.metric("Confiance ML",   f"{confiance}%")
        col_r3.metric("Audience totale", f"{audience:,}")

        # Graphique probabilites
        df_prob = pd.DataFrame({
            "Niveau": modeles["encoder"].classes_,
            "Proba":  [round(p*100, 1) for p in probas],
        }).sort_values("Proba", ascending=True)

        fig_prob = go.Figure(go.Bar(
            x=df_prob["Proba"],
            y=df_prob["Niveau"],
            orientation="h",
            marker_color=[COULEURS_NIVEAU.get(n, "#6b7280")
                          for n in df_prob["Niveau"]],
            text=[f"{p}%" for p in df_prob["Proba"]],
            textposition="outside",
        ))
        fig_prob.update_layout(
            title="Probabilites par niveau",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8",
            height=250,
            margin=dict(l=0, r=60, t=40, b=0),
            xaxis=dict(range=[0, 110], gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b"),
        )
        st.plotly_chart(fig_prob, use_container_width=True)

        # Conseils selon niveau
        conseils = {
            "Mega":  ("Mega-influenceur", "Ideal pour grandes campagnes nationales", "> 10 000 TND"),
            "Macro": ("Macro-influenceur", "Bon rapport qualite/prix", "2 000 - 10 000 TND"),
            "Micro": ("Micro-influenceur", "Excellent engagement niche", "500 - 2 000 TND"),
            "Nano":  ("Nano-influenceur",  "Communaute tres engagee", "100 - 500 TND"),
        }
        titre_c, desc_c, budget_c = conseils.get(niveau, ("","",""))
        couleur_c = COULEURS_NIVEAU.get(niveau, "#6b7280")

        st.markdown(f"""
        <div style="background:#1e293b;border:1px solid {couleur_c};
                    border-radius:12px;padding:20px;margin-top:16px">
            <div style="font-size:20px;font-weight:700;color:{couleur_c}">
                {titre_c}
            </div>
            <div style="color:#94a3b8;margin-top:6px">{desc_c}</div>
            <div style="color:#e2e8f0;margin-top:8px;font-weight:600">
                Budget recommande : {budget_c}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Alerte fake
        if ratio_ff > 2.0:
            st.markdown(f"""
            <div class="warning-box">
                Compte suspect : ratio following/followers = {ratio_ff:.1f}
                (seuil normal < 0.5). Possible achat de followers.
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  PAGE 3 : RECOMMANDATION MARQUES
# ══════════════════════════════════════════════════════

elif page == "Recommandation marques":
    st.title("Recommandation pour marques")
    st.markdown("Trouvez les meilleurs influenceurs pour votre campagne")

    if df.empty:
        st.error("Dataset introuvable")
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
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
        secteur = st.selectbox("Secteur", list(SECTEURS.keys()))

    with col2:
        niveau_opts = ["Tous niveaux", "Mega", "Macro", "Micro", "Nano"]
        niveau_f = st.selectbox("Niveau", niveau_opts)

    with col3:
        top_n = st.slider("Nombre de resultats", 3, 20, 5)

    if st.button("Rechercher", use_container_width=True):
        df_rec = df.copy()
        cats = SECTEURS.get(secteur, [])
        if cats:
            df_rec = df_rec[df_rec["categorie"].isin(cats)]
        if niveau_f != "Tous niveaux":
            df_rec = df_rec[df_rec["niveau"] == niveau_f]

        df_rec = df_rec.sort_values("score_influence", ascending=False)

        st.markdown("---")
        st.markdown(f"**{len(df_rec)} influenceurs** correspondent — affichage des {min(top_n, len(df_rec))} meilleurs")
        st.markdown("")

        if df_rec.empty:
            st.warning("Aucun influenceur trouve pour ces criteres.")
        else:
            for i, (_, row) in enumerate(df_rec.head(top_n).iterrows(), 1):
                niv      = row.get("niveau", "")
                badge_c  = f"badge-{niv.lower()}"
                score    = int(row.get("score_influence", 0))
                ig_f_r   = int(row.get("instagram_followers", 0))
                tk_f_r   = int(row.get("tiktok_followers", 0))
                yt_s_r   = int(row.get("youtube_subscribers", 0))
                audience = int(row.get("audience_totale", ig_f_r + tk_f_r + yt_s_r))

                # Barre de score
                barre_w = score

                st.markdown(f"""
                <div class="influencer-card">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start">
                        <div>
                            <span style="color:#64748b;font-size:13px">#{i}</span>
                            <span style="font-size:16px;font-weight:700;color:#e2e8f0;margin-left:8px">
                                @{row["username"]}
                            </span>
                            <span style="color:#64748b;font-size:13px;margin-left:6px">
                                — {row.get("nom","")}
                            </span>
                        </div>
                        <div style="display:flex;gap:8px;align-items:center">
                            <span class="{badge_c}">{niv}</span>
                            <span style="color:#a5b4fc;font-weight:700;font-size:15px">
                                {score}/100
                            </span>
                        </div>
                    </div>
                    <div style="margin-top:10px;display:flex;gap:20px;flex-wrap:wrap">
                        <span style="color:#94a3b8;font-size:13px">
                            Categorie: <b style="color:#e2e8f0">{row.get("categorie","")}</b>
                        </span>
                        <span style="color:#94a3b8;font-size:13px">
                            Instagram: <b style="color:#e2e8f0">{ig_f_r:,}</b>
                        </span>
                        <span style="color:#94a3b8;font-size:13px">
                            TikTok: <b style="color:#e2e8f0">{tk_f_r:,}</b>
                        </span>
                        <span style="color:#94a3b8;font-size:13px">
                            YouTube: <b style="color:#e2e8f0">{yt_s_r:,}</b>
                        </span>
                        <span style="color:#94a3b8;font-size:13px">
                            Audience: <b style="color:#e2e8f0">{audience:,}</b>
                        </span>
                    </div>
                    <div style="margin-top:10px;background:#0f172a;border-radius:4px;height:4px">
                        <div style="width:{barre_w}%;height:4px;border-radius:4px;
                                    background:linear-gradient(90deg,#6366f1,#8b5cf6)">
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Telechargement
        csv = df_rec.head(top_n).to_csv(index=False)
        st.download_button(
            "Telecharger les resultats (CSV)",
            csv,
            f"recommandations_{secteur.replace('/','-')}.csv",
            "text/csv",
        )


# ══════════════════════════════════════════════════════
#  PAGE 4 : CLASSEMENT COMPLET
# ══════════════════════════════════════════════════════

elif page == "Classement complet":
    st.title("Classement complet")

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
    st.markdown(f"**{len(df_f)} influenceurs**")

    cols_afficher = ["username", "nom", "categorie", "niveau",
                     "instagram_followers", "tiktok_followers",
                     "youtube_subscribers", "score_influence"]
    cols_afficher = [c for c in cols_afficher if c in df_f.columns]

    st.dataframe(
        df_f[cols_afficher].reset_index(drop=True),
        use_container_width=True,
        height=500,
        column_config={
            "instagram_followers": st.column_config.NumberColumn(
                "IG Followers", format="%d"
            ),
            "tiktok_followers": st.column_config.NumberColumn(
                "TK Followers", format="%d"
            ),
            "youtube_subscribers": st.column_config.NumberColumn(
                "YT Abonnes", format="%d"
            ),
            "score_influence": st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=100
            ),
        }
    )

    csv = df_f[cols_afficher].to_csv(index=False)
    st.download_button("Telecharger CSV", csv, "classement.csv", "text/csv")