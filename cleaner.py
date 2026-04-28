"""
UTILS — Cleaner & Merger
Fusionne les DataFrames de toutes les sources,
nettoie, déduplique, calcule les features finales.
"""

import re
import numpy as np
import pandas as pd
import logging

log = logging.getLogger(__name__)

# ─── MAPPING CATÉGORIES → Labels standardisés ─────────────────────────────────
CATEGORY_NORMALIZE = {
    # Mode / Fashion
    "fashion": "Mode", "mode": "Mode", "style": "Mode",
    "modest fashion": "Mode", "hijab": "Mode", "vêtements": "Mode",
    "luxe": "Mode", "luxury": "Mode",

    # Beauté / Makeup
    "beauty": "Beauté", "beauté": "Beauté", "makeup": "Beauté",
    "skincare": "Beauté", "cosmetics": "Beauté",

    # Food / Cuisine
    "food": "Food", "cuisine": "Food", "cooking": "Food",
    "recette": "Food", "pâtisserie": "Food", "gastronomie": "Food",
    "restaurant": "Food",

    # Sport
    "sport": "Sport", "sports": "Sport", "football": "Sport",
    "tennis": "Sport", "basketball": "Sport", "athlétisme": "Sport",
    "cricket": "Sport",

    # Fitness / Santé
    "fitness": "Fitness / Santé", "health": "Fitness / Santé",
    "gym": "Fitness / Santé", "musculation": "Fitness / Santé",
    "wellbeing": "Fitness / Santé", "crossfit": "Fitness / Santé",

    # Musique
    "music": "Musique", "musique": "Musique", "artiste": "Musique",
    "chanteuse": "Musique", "rap": "Musique", "chanteur": "Musique",

    # Entertainment / Acteurs
    "entertainment": "Entertainment", "actrice": "Entertainment",
    "acteur": "Entertainment", "cinema": "Entertainment",
    "tv": "Entertainment", "télévision": "Entertainment",

    # Humour
    "comedy": "Humour", "humour": "Humour", "comique": "Humour",
    "humor": "Humour",

    # Voyage / Tourisme
    "travel": "Voyage", "voyage": "Voyage", "tourisme": "Voyage",
    "tourism": "Voyage",

    # Tech / Digital
    "technology": "Tech", "tech": "Tech", "digital": "Tech",
    "smartphones": "Tech", "gadgets": "Tech", "gaming": "Gaming",

    # Gaming
    "gaming": "Gaming", "games": "Gaming", "jeux vidéo": "Gaming",
    "esport": "Gaming",

    # Business / Entrepreneuriat
    "business": "Business", "entrepreneuriat": "Business",
    "startup": "Business", "marketing": "Business",
    "motivation": "Business",

    # Education / Science
    "education": "Education", "science": "Education",
    "nature": "Education",

    # Art / Créatif
    "art": "Art / Créatif", "photography": "Art / Créatif",
    "design": "Art / Créatif",

    # Famille / Maternité
    "family": "Famille", "famille": "Famille",
    "maternité": "Famille", "vlog": "Famille",

    # Activisme
    "activism": "Activisme", "activisme": "Activisme",
    "politique": "Activisme", "social": "Activisme",

    # Actualités
    "news": "Actualités", "actualités": "Actualités",
    "journalism": "Actualités",
}


def _normalize_category(cat: str) -> str:
    """Normalise une catégorie brute vers les labels du projet."""
    if not cat or pd.isna(cat):
        return "Lifestyle"
    cat_lower = cat.lower().strip()
    for key, value in CATEGORY_NORMALIZE.items():
        if key in cat_lower:
            return value
    return cat.strip().title() if cat.strip() else "Lifestyle"


def _compute_tier(followers: float) -> str:
    """Classifie l'influenceur selon son nombre de followers."""
    if followers >= 1_000_000:
        return "Mega (>1M)"
    elif followers >= 100_000:
        return "Macro (100K-1M)"
    elif followers >= 10_000:
        return "Micro (10K-100K)"
    elif followers >= 1_000:
        return "Nano (1K-10K)"
    else:
        return "Inconnu"


def _compute_reach_score(row: pd.Series) -> float:
    """
    Calcule un reach_score normalisé [0-100] basé sur :
    - followers (poids 50%)
    - engagement_rate (poids 30%)
    - avg_likes (poids 20%)
    """
    # Normalisation log pour les followers (évite l'écrasement par les mega)
    f_score = min(np.log1p(row.get("followers", 0)) / np.log1p(20_000_000) * 100, 100)
    e_score = min(row.get("engagement_rate", 0) / 15.0 * 100, 100)  # max 15% engagement
    l_score = min(np.log1p(row.get("avg_likes", 0)) / np.log1p(1_000_000) * 100, 100)

    return round(0.50 * f_score + 0.30 * e_score + 0.20 * l_score, 2)


def _detect_gender(username: str, nom: str = "") -> str:
    """Heuristique simple basée sur les prénoms tunisiens courants."""
    text = (username + " " + nom).lower()
    feminine = ["dorra", "douha", "oumaima", "hend", "manel", "rym",
                "fatma", "chahrazed", "nawres", "yousra", "roua",
                "leila", "eya", "khouloud", "samara", "lina",
                "rawdha", "nadia", "fatima", "priyanka"]
    masculine = ["cristiano", "samy", "skander", "ali", "7chich",
                 "houssem", "walid", "midox", "ayoub", "ryan",
                 "mokhtar", "salim", "hafedh", "sportif", "hagani",
                 "issam", "david", "chef"]
    for name in feminine:
        if name in text:
            return "F"
    for name in masculine:
        if name in text:
            return "M"
    return "Inconnu"


def clean_and_merge(dataframes: list) -> pd.DataFrame:
    """
    Fusionne et nettoie les DataFrames de toutes les sources scraper.
    
    Pipeline :
    1. Concaténation de tous les DFs
    2. Normalisation des colonnes
    3. Déduplication par username (garde les meilleures valeurs)
    4. Calcul des features dérivées
    5. Tri par reach_score
    6. Ajout du rank final
    
    Args:
        dataframes: Liste de DataFrames de chaque scraper
    
    Returns:
        DataFrame final propre avec toutes les features
    """
    log.info("   Concaténation des sources...")

    # Colonnes attendues
    EXPECTED_COLS = [
        "source", "username", "followers", "engagement_rate",
        "avg_likes", "categorie"
    ]

    # Normaliser chaque DF avant merge
    normalized = []
    for df in dataframes:
        if df is None or df.empty:
            continue
        # Ajouter les colonnes manquantes avec valeur par défaut
        for col in EXPECTED_COLS:
            if col not in df.columns:
                df[col] = None
        normalized.append(df[df.columns.intersection(
            EXPECTED_COLS + [c for c in df.columns if c not in EXPECTED_COLS]
        )])

    if not normalized:
        return pd.DataFrame()

    df_all = pd.concat(normalized, ignore_index=True)
    log.info(f"   Total brut : {len(df_all)} entrées")

    # ── Nettoyage des usernames ────────────────────────────────────────────
    df_all["username"] = (
        df_all["username"]
        .astype(str)
        .str.strip()
        .str.lstrip("@")
        .str.lower()
        .str.replace(r"\s+", "", regex=True)
    )
    df_all = df_all[df_all["username"].str.len() > 0]
    df_all = df_all[df_all["username"] != "nan"]

    # ── Conversion numérique ──────────────────────────────────────────────
    df_all["followers"]       = pd.to_numeric(df_all["followers"], errors="coerce").fillna(0)
    df_all["engagement_rate"] = pd.to_numeric(df_all["engagement_rate"], errors="coerce").fillna(0)
    df_all["avg_likes"]       = pd.to_numeric(df_all.get("avg_likes", 0), errors="coerce").fillna(0)

    # ── Déduplication : garder la ligne avec le max followers par username ──
    log.info("   Déduplication par username...")
    df_dedup = (
        df_all
        .sort_values("followers", ascending=False)
        .groupby("username", as_index=False)
        .first()
    )

    # Fusionner les champs complémentaires (engagement depuis source différente)
    eng_best = (
        df_all[df_all["engagement_rate"] > 0]
        .sort_values("engagement_rate", ascending=False)
        .groupby("username")["engagement_rate"]
        .first()
        .reset_index()
    )
    df_dedup = df_dedup.merge(eng_best, on="username", how="left", suffixes=("", "_best"))
    df_dedup["engagement_rate"] = df_dedup["engagement_rate_best"].combine_first(
        df_dedup["engagement_rate"]
    )
    df_dedup.drop(columns=["engagement_rate_best"], inplace=True, errors="ignore")

    # ── Features dérivées ─────────────────────────────────────────────────
    log.info("   Calcul des features dérivées...")

    df_dedup["followers_M"]      = (df_dedup["followers"] / 1_000_000).round(3)
    df_dedup["categorie"]        = df_dedup["categorie"].apply(_normalize_category)
    df_dedup["tier"]             = df_dedup["followers"].apply(_compute_tier)
    df_dedup["reach_score"]      = df_dedup.apply(_compute_reach_score, axis=1)
    df_dedup["genre"]            = df_dedup["username"].apply(_detect_gender)

    # Estimer avg_likes si absent (approximation : followers * engagement / 100)
    mask = df_dedup["avg_likes"] == 0
    df_dedup.loc[mask, "avg_likes"] = (
        df_dedup.loc[mask, "followers"] * df_dedup.loc[mask, "engagement_rate"] / 100
    ).round(0)

    df_dedup["avg_comments"] = (df_dedup["avg_likes"] * 0.04).round(0).astype(int)

    # ── Filtres qualité ────────────────────────────────────────────────────
    # Garder seulement les profils avec au moins 1000 followers
    df_clean = df_dedup[df_dedup["followers"] >= 1_000].copy()

    # ── Tri et ranking final ───────────────────────────────────────────────
    df_clean.sort_values("followers", ascending=False, inplace=True)
    df_clean.reset_index(drop=True, inplace=True)
    df_clean.index += 1
    df_clean.insert(0, "rank", df_clean.index)

    # ── Sélection et ordre des colonnes finales ────────────────────────────
    FINAL_COLS = [
        "rank", "username", "followers", "followers_M",
        "engagement_rate", "avg_likes", "avg_comments",
        "categorie", "genre", "tier", "reach_score",
    ]
    # Ajouter colonnes optionnelles si présentes
    for optional in ["favikon_score", "audience_score", "grade",
                      "monthly_growth_avg", "nb_posts", "following"]:
        if optional in df_clean.columns:
            FINAL_COLS.append(optional)

    existing_cols = [c for c in FINAL_COLS if c in df_clean.columns]
    df_final = df_clean[existing_cols].copy()

    log.info(f"   ✅ Dataset final : {len(df_final)} influenceurs | "
             f"{df_final['categorie'].nunique()} catégories")

    return df_final
