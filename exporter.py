"""
UTILS — Exporter
Sauvegarde le dataset final en CSV et Excel formaté.
"""

import os
import logging
import pandas as pd

log = logging.getLogger(__name__)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _style_excel(writer: pd.ExcelWriter, df: pd.DataFrame, sheet_name: str) -> None:
    """Applique une mise en forme Excel professionnelle."""
    workbook  = writer.book
    worksheet = writer.sheets[sheet_name]

    # ── Formats ────────────────────────────────────────────────────────────
    header_fmt = workbook.add_format({
        "bold":       True,
        "bg_color":   "#1E3A5F",   # Bleu foncé
        "font_color": "#FFFFFF",
        "border":     1,
        "align":      "center",
        "valign":     "vcenter",
        "text_wrap":  True,
    })
    even_fmt = workbook.add_format({
        "bg_color": "#EBF5FB",
        "border":   1,
        "align":    "center",
        "valign":   "vcenter",
    })
    odd_fmt = workbook.add_format({
        "bg_color": "#FFFFFF",
        "border":   1,
        "align":    "center",
        "valign":   "vcenter",
    })
    number_fmt = workbook.add_format({
        "num_format": "#,##0",
        "border":     1,
        "align":      "center",
    })
    percent_fmt = workbook.add_format({
        "num_format": "0.00%",
        "border":     1,
        "align":      "center",
    })

    # ── En-têtes ───────────────────────────────────────────────────────────
    for col_num, col_name in enumerate(df.columns):
        worksheet.write(0, col_num, col_name, header_fmt)

    # ── Données avec alternance de couleurs ───────────────────────────────
    for row_num in range(len(df)):
        fmt = even_fmt if row_num % 2 == 0 else odd_fmt
        for col_num, col_name in enumerate(df.columns):
            val = df.iloc[row_num, col_num]
            if pd.isna(val):
                val = ""
            worksheet.write(row_num + 1, col_num, val, fmt)

    # ── Largeurs de colonnes ───────────────────────────────────────────────
    col_widths = {
        "rank":              6,
        "username":          22,
        "followers":         14,
        "followers_M":       12,
        "engagement_rate":   16,
        "avg_likes":         14,
        "avg_comments":      14,
        "categorie":         20,
        "genre":             8,
        "tier":              18,
        "reach_score":       12,
        "favikon_score":     14,
        "audience_score":    14,
        "grade":             8,
        "monthly_growth_avg": 18,
        "nb_posts":          10,
        "following":         12,
    }
    for col_num, col_name in enumerate(df.columns):
        width = col_widths.get(col_name, 15)
        worksheet.set_column(col_num, col_num, width)

    # ── Freeze header row ─────────────────────────────────────────────────
    worksheet.freeze_panes(1, 0)

    # ── Filtre automatique ────────────────────────────────────────────────
    worksheet.autofilter(0, 0, len(df), len(df.columns) - 1)


def _add_stats_sheet(writer: pd.ExcelWriter, df: pd.DataFrame) -> None:
    """Ajoute une feuille de statistiques résumées."""
    stats_data = []

    # Stats globales
    stats_data.append({"Métrique": "Total influenceurs", "Valeur": len(df)})
    stats_data.append({"Métrique": "Mega (>1M followers)", "Valeur": (df["tier"] == "Mega (>1M)").sum()})
    stats_data.append({"Métrique": "Macro (100K-1M)",      "Valeur": (df["tier"] == "Macro (100K-1M)").sum()})
    stats_data.append({"Métrique": "Micro (10K-100K)",     "Valeur": (df["tier"] == "Micro (10K-100K)").sum()})
    stats_data.append({"Métrique": "Avg engagement rate",  "Valeur": round(df["engagement_rate"].mean(), 2)})
    stats_data.append({"Métrique": "Max followers (M)",    "Valeur": round(df["followers_M"].max(), 2)})
    stats_data.append({"Métrique": "Profils féminins",     "Valeur": (df["genre"] == "F").sum()})
    stats_data.append({"Métrique": "Profils masculins",    "Valeur": (df["genre"] == "M").sum()})

    # Répartition par catégorie
    cat_counts = df["categorie"].value_counts().reset_index()
    cat_counts.columns = ["Catégorie", "Nb influenceurs"]

    df_stats = pd.DataFrame(stats_data)
    df_stats.to_excel(writer, sheet_name="Stats", index=False)
    cat_counts.to_excel(writer, sheet_name="Stats", startrow=len(df_stats) + 2, index=False)

    # Mise en forme basique
    workbook  = writer.book
    worksheet = writer.sheets["Stats"]
    header_fmt = workbook.add_format({
        "bold": True, "bg_color": "#1E3A5F",
        "font_color": "#FFFFFF", "border": 1
    })
    worksheet.write(0, 0, "Métrique", header_fmt)
    worksheet.write(0, 1, "Valeur",   header_fmt)
    worksheet.set_column(0, 0, 28)
    worksheet.set_column(1, 1, 18)


def export_dataset(df: pd.DataFrame) -> None:
    """
    Sauvegarde le dataset final en :
    - CSV  : outputs/top_influenceurs_tunisiens.csv
    - XLSX : outputs/top_influenceurs_tunisiens.xlsx
    
    Args:
        df: DataFrame final nettoyé
    """
    csv_path  = os.path.join(OUTPUT_DIR, "top_influenceurs_tunisiens.csv")
    xlsx_path = os.path.join(OUTPUT_DIR, "top_influenceurs_tunisiens.xlsx")

    # ── CSV ───────────────────────────────────────────────────────────────
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    log.info(f"   ✅ CSV  : {csv_path}  ({len(df)} lignes)")

    # ── XLSX ──────────────────────────────────────────────────────────────
    try:
        import xlsxwriter  # noqa
        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="Influenceurs", index=False)
            _style_excel(writer, df, "Influenceurs")
            _add_stats_sheet(writer, df)
        log.info(f"   ✅ XLSX : {xlsx_path}  (avec mise en forme)")
    except ImportError:
        log.warning("   ⚠️  xlsxwriter non installé, export XLSX ignoré")
        log.warning("       pip install xlsxwriter")

    # ── Aperçu console ────────────────────────────────────────────────────
    log.info("\n   ── APERÇU TOP 10 ──────────────────────────────────")
    log.info("\n" + df.head(10).to_string(index=False))
