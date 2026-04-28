import json
import os
import re
import base64
import streamlit as st
from datetime import datetime

try:
    import bcrypt
    _BCRYPT_OK = True
except ImportError:
    import hashlib
    _BCRYPT_OK = False

# ── Chemin fichier users ──────────────────────────────────────────
_ROOT       = os.path.dirname(os.path.abspath(__file__))
_USERS_FILE = os.path.join(_ROOT, "data",   "users.json")
_LOGO_SVG   = os.path.join(_ROOT, "assets", "logo_horizontal.svg")
_LOGO_PNG   = os.path.join(_ROOT, "assets", "logo_horizontal.png")


def _load_logo_content() -> str:
    """
    Charge le logo et retourne son contenu.
    Priorité : SVG > PNG > fallback texte.
    """
    # 1. Essayer le SVG (qualité parfaite)
    for path in [
        _LOGO_SVG,
        os.path.join(_ROOT, "logo_horizontal.svg"),
        os.path.join(_ROOT, "assets", "logo.svg"),
    ]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return ("svg", f.read())
    
    # 2. Essayer le PNG (fallback)
    for path in [
        _LOGO_PNG,
        os.path.join(_ROOT, "logo_horizontal.png"),
        os.path.join(_ROOT, "assets", "logo.png"),
    ]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            return ("png", b64)
    
    # 3. Aucun logo trouvé
    return ("none", "")


# ════════════════════════════════════════════════════════════════
#  HELPERS STOCKAGE
# ════════════════════════════════════════════════════════════════

def _load_users() -> dict:
    if not os.path.exists(_USERS_FILE):
        return {}
    try:
        with open(_USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_users(users: dict) -> None:
    os.makedirs(os.path.dirname(_USERS_FILE), exist_ok=True)
    with open(_USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)


# ════════════════════════════════════════════════════════════════
#  HELPERS MOT DE PASSE
# ════════════════════════════════════════════════════════════════

def _hash_password(password: str) -> str:
    if _BCRYPT_OK:
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    return hashlib.sha256(password.encode()).hexdigest()


def _check_password(password: str, hashed: str) -> bool:
    if _BCRYPT_OK:
        try:
            return bcrypt.checkpw(password.encode(), hashed.encode())
        except Exception:
            return False
    return hashlib.sha256(password.encode()).hexdigest() == hashed


# ════════════════════════════════════════════════════════════════
#  VALIDATION
# ════════════════════════════════════════════════════════════════

def _validate_email(email: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email))


def _validate_password(pwd: str) -> list[str]:
    errors = []
    if len(pwd) < 8:
        errors.append("Au moins 8 caractères")
    if not re.search(r"[A-Z]", pwd):
        errors.append("Au moins une majuscule")
    if not re.search(r"\d", pwd):
        errors.append("Au moins un chiffre")
    return errors


# ════════════════════════════════════════════════════════════════
#  API PUBLIQUE
# ════════════════════════════════════════════════════════════════

def register_user(username: str, email: str, password: str) -> tuple[bool, str]:
    username = username.strip()
    email    = email.strip().lower()

    if not username or len(username) < 3:
        return False, "Le nom d'utilisateur doit contenir au moins 3 caractères."
    if not re.match(r"^[a-zA-Z0-9_]+$", username):
        return False, "Le nom d'utilisateur ne peut contenir que des lettres, chiffres et _."
    if not _validate_email(email):
        return False, "Adresse e-mail invalide."
    pwd_errors = _validate_password(password)
    if pwd_errors:
        return False, "Mot de passe invalide : " + " · ".join(pwd_errors)

    users = _load_users()

    if username.lower() in {u.lower() for u in users}:
        return False, "Ce nom d'utilisateur est déjà pris."
    if email in {u.get("email", "") for u in users.values()}:
        return False, "Cette adresse e-mail est déjà utilisée."

    users[username] = {
        "email":      email,
        "password":   _hash_password(password),
        "created_at": datetime.now().isoformat(),
        "role":       "user",
    }
    _save_users(users)
    return True, ""


def login_user(username: str, password: str) -> tuple[bool, str]:
    username = username.strip()
    users    = _load_users()

    match = next((u for u in users if u.lower() == username.lower()), None)
    if not match:
        return False, "Nom d'utilisateur ou mot de passe incorrect."
    if not _check_password(password, users[match]["password"]):
        return False, "Nom d'utilisateur ou mot de passe incorrect."

    return True, match


def is_logged_in() -> bool:
    return st.session_state.get("auth_logged_in", False)


def current_user() -> str:
    return st.session_state.get("auth_username", "")


def logout():
    for key in ["auth_logged_in", "auth_username", "auth_email"]:
        st.session_state.pop(key, None)
    st.rerun()


# ════════════════════════════════════════════════════════════════
#  CSS SALMVERSE — PAGE AUTH
# ════════════════════════════════════════════════════════════════

_AUTH_CSS = """
/* Centrage logo */
[data-testid="stImage"] > div,
[data-testid="stImage"] > img,
.stImage {
    display: flex !important;
    justify-content: center !important;
    text-align: center !important;
    margin: 0 auto !important;
}
<style>
/* === Google Fonts === */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Sora:wght@400;600;700;800&display=swap');

/* === Palette Salmverse === */
:root {
    --dark-bg: #250e2c;
    --violet-mid: #837ab6;
    --violet-tender: #9d85b6;
    --accent-mauve: #cc8db3;
    --cta-pink: #f6a5c0;
    --bg-light: #f7c2ca;
    --white: #FFFFFF;
}

[data-testid="stImage"] img {
    margin: 0 auto !important;
    display: block !important;
}
/* === Masquer la sidebar === */
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* === Fond de la page auth === */
.stApp {
background: linear-gradient(160deg, #4e2b6e 0%, #5e3485 40%, #3f2a78 100%) !important;
/* === Conteneur principal === */
.stMainBlockContainer {
    padding-top: 2rem !important;
}

/* === Inputs style Salmverse === */
[data-testid="stTextInput"] input {
    background: rgba(240, 230, 255, 0.92) !important;
    border: 1.5px solid rgba(157, 133, 182, 0.45) !important;
    border-radius: 12px !important;
    color: #2a1040 !important;
    caret-color: #6d4fa0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    padding: 12px 16px !important;
    transition: all 0.3s ease !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #8a5cb8 !important;
    box-shadow: 0 0 0 4px rgba(138, 92, 184, 0.2) !important;
    background: rgba(248, 242, 255, 0.97) !important;
}
[data-testid="stTextInput"] input::placeholder {
    color: #9880b0 !important;
}
[data-testid="stTextInput"] label {
    color: rgba(246, 165, 192, 0.85) !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-family: 'Inter', sans-serif !important;
}
div[data-testid="stTextInput"] label p {
    color: rgba(246, 165, 192, 0.85) !important;
    font-size: 12px !important;
    font-weight: 600 !important;
}


[data-testid="stFormSubmitButton"] button {
    background: #7b4fa6 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 30px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 12px 24px !important;
    width: 100% !important;
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.25) !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFormSubmitButton"] button:hover {
    background: #9060c0 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3) !important;
}

.stButton > button[kind="primary"] {
    background: #7b4fa6 !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2) !important;
}

.stButton > button[kind="secondary"] {
    background: #5a3680 !important;
    color: rgba(255, 255, 255, 0.85) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.15) !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #7b4fa6 !important;
    color: #ffffff !important;
    border-color: rgba(255, 255, 255, 0.3) !important;
}
/* === Alertes === */
.stAlert {
    background: rgba(37, 14, 44, 0.8) !important;
    border: 1.5px solid rgba(204, 141, 179, 0.3) !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    backdrop-filter: blur(10px) !important;
}
.stAlert [data-testid="stMarkdownContainer"] p {
    color: #f6a5c0 !important;
}

/* === Barre de force du mot de passe === */
.pwd-bar-wrap {
    background: rgba(255, 255, 255, 0.06);
    border-radius: 6px;
    height: 5px;
    margin: 8px 0 6px;
    overflow: hidden;
}
.pwd-bar {
    height: 5px;
    border-radius: 6px;
    transition: width 0.4s ease, background 0.4s ease;
}
.pwd-rule {
    font-size: 12px;
    padding: 3px 0;
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'Inter', sans-serif;
}
.pwd-ok  { color: #4ade80; }
.pwd-bad { color: rgba(204, 141, 179, 0.4); }

/* === Logo card === */
.logo-card {
    text-align: center;
    background: linear-gradient(160deg, rgba(255,255,255,0.15) 0%, rgba(246,165,192,0.12) 100%);
    border: 1.5px solid rgba(246,165,192,0.35);
    border-radius: 20px;
    padding: 32px 24px 22px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1);
}
.logo-card .glow-orb {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -58%);
    width: 240px;
    height: 200px;
    background: radial-gradient(ellipse, rgba(246, 165, 192, 0.2) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}
.logo-card .decor-top-right {
    position: absolute;
    top: -30px;
    right: -30px;
    width: 100px;
    height: 100px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(204, 141, 179, 0.15) 0%, transparent 70%);
    pointer-events: none;
}
.logo-card .decor-bottom-left {
    position: absolute;
    bottom: -20px;
    left: -20px;
    width: 80px;
    height: 80px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(131, 122, 182, 0.12) 0%, transparent 70%);
    pointer-events: none;
}
.logo-card .separator {
    width: 48px;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(204, 141, 179, 0.6), transparent);
    margin: 16px auto 10px;
    border-radius: 2px;
}
.logo-card .subtitle {
    font-size: 10.5px;
    color: rgba(246, 165, 192, 0.5);
    letter-spacing: 0.16em;
    text-transform: uppercase;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
}

/* === Titre de section === */
.section-title {
    font-family: 'Sora', sans-serif;
    font-size: 16px;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 16px;
}

/* === Footer === */
.auth-footer {
    text-align: center;
    margin-top: 20px;
    font-size: 11px;
    color: rgba(204, 141, 179, 0.35);
    font-family: 'Inter', sans-serif;
    line-height: 1.6;
}
</style>
"""


# ════════════════════════════════════════════════════════════════
#  RENDU PAGE LOGIN / REGISTER
# ════════════════════════════════════════════════════════════════

def render_auth_page():
    st.markdown(_AUTH_CSS, unsafe_allow_html=True)

    st.markdown("<div style='height:36px'></div>", unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.8, 1])

    with col:
        # ── LOGO HEADER CARD ─────────────────────────────────────
        logo_svg_path = os.path.join(_ROOT, "assets", "logo_horizontal.svg")
        
        # Chercher le SVG
        svg_found = False
        for path in [
            logo_svg_path,
            os.path.join(_ROOT, "logo_horizontal.svg"),
        ]:
            if os.path.exists(path):
                svg_found = True
                logo_path = path
                break
        
        # Carte du logo
        st.markdown("""
        <div class="logo-card">
            <div class="glow-orb"></div>
            <div class="decor-top-right"></div>
            <div class="decor-bottom-left"></div>
        """, unsafe_allow_html=True)
        
        if svg_found:
             col_l, col_img, col_r = st.columns([1, 2, 1])
             with col_img:
                 st.image(logo_path, use_container_width=True)
                 
                 
                 
            
            # Afficher le SVG avec st.image() — supporte le SVG natif
            
        else:
            # Fallback
            st.markdown("""
            <div style="font-size:48px;position:relative;z-index:1;color:#f6a5c0;text-align:center;">✨</div>
            <div style="font-family:'Sora',sans-serif;font-size:20px;font-weight:700;
                        color:#ffffff;position:relative;z-index:1;margin-top:4px;text-align:center;">
                Influence TN
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="separator"></div>
            <div class="subtitle">Plateforme d'analyse &nbsp;·&nbsp; 2025</div>
        </div>
        """, unsafe_allow_html=True)
     

        # ── ONGLETS LOGIN / REGISTER ──────────────────────────────
        if "auth_tab" not in st.session_state:
            st.session_state["auth_tab"] = "login"

        tab_col1, tab_col2 = st.columns(2)
        with tab_col1:
            if st.button("🔑  Connexion",
                         use_container_width=True,
                         type="primary" if st.session_state["auth_tab"] == "login" else "secondary",
                         key="btn_login_tab"):
                st.session_state["auth_tab"] = "login"
                st.rerun()
        with tab_col2:
            if st.button("✨  Créer un compte",
                         use_container_width=True,
                         type="primary" if st.session_state["auth_tab"] == "register" else "secondary",
                         key="btn_register_tab"):
                st.session_state["auth_tab"] = "register"
                st.rerun()

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # ── LOGIN ────────────────────────────────────────────────
        if st.session_state["auth_tab"] == "login":
            with st.form("form_login", clear_on_submit=False):
                st.markdown(
                    '<div class="section-title">Bienvenue 🎉</div>',
                    unsafe_allow_html=True,
                )
                username = st.text_input("Nom d'utilisateur", placeholder="votre_username")
                password = st.text_input("Mot de passe", type="password", placeholder="••••••••")
                submitted = st.form_submit_button("Se connecter →", use_container_width=True)

            if submitted:
                if not username or not password:
                    st.error("Veuillez remplir tous les champs.")
                else:
                    ok, result = login_user(username, password)
                    if ok:
                        st.session_state["auth_logged_in"] = True
                        st.session_state["auth_username"]  = result
                        users = _load_users()
                        st.session_state["auth_email"] = users.get(result, {}).get("email", "")
                        st.success(f"✅ Connexion réussie ! Bienvenue **{result}**")
                        st.balloons()
                        import time; time.sleep(0.8)
                        st.rerun()
                    else:
                        st.error(f"❌ {result}")

        # ── REGISTER ─────────────────────────────────────────────
        else:
            with st.form("form_register", clear_on_submit=True):
                st.markdown(
                    '<div class="section-title">Créer un compte ✨</div>',
                    unsafe_allow_html=True,
                )
                username = st.text_input("Nom d'utilisateur", placeholder="ex: ahmed_boudriga")
                email    = st.text_input("Adresse e-mail", placeholder="email@exemple.com")
                password = st.text_input("Mot de passe", type="password",
                                         placeholder="Min. 8 car. · 1 maj. · 1 chiffre",
                                         help="Min. 8 caractères · 1 majuscule · 1 chiffre")
                confirm  = st.text_input("Confirmer le mot de passe", type="password",
                                         placeholder="••••••••")
                submitted = st.form_submit_button("Créer mon compte →", use_container_width=True)

            # Indicateur force mot de passe
            if password:
                rules = [
                    (len(password) >= 8,        "8 caractères minimum"),
                    (bool(__import__('re').search(r"[A-Z]", password)), "1 majuscule"),
                    (bool(__import__('re').search(r"\d",   password)), "1 chiffre"),
                ]
                score  = sum(ok for ok, _ in rules)
                colors = ["#ef4444", "#f97316", "#22c55e"]
                widths = ["33%", "66%", "100%"]
                labels = ["Faible", "Moyen", "Fort"]
                st.markdown(f"""
                <div style="margin:-8px 0 8px">
                  <div class="pwd-bar-wrap">
                    <div class="pwd-bar"
                         style="width:{widths[score-1] if score else '0%'};
                                background:{colors[score-1] if score else '#ef4444'}">
                    </div>
                  </div>
                  <div style="font-size:11px;color:{colors[score-1] if score else '#6b7280'};font-family:'Inter',sans-serif;">
                    {labels[score-1] if score else ''}
                  </div>
                  {"".join(
                      f'<div class="pwd-rule {"pwd-ok" if ok else "pwd-bad"}">'
                      f'{"✓" if ok else "○"} {label}</div>'
                      for ok, label in rules
                  )}
                </div>
                """, unsafe_allow_html=True)

            if submitted:
                if not all([username, email, password, confirm]):
                    st.error("Veuillez remplir tous les champs.")
                elif password != confirm:
                    st.error("❌ Les mots de passe ne correspondent pas.")
                else:
                    ok, msg = register_user(username, email, password)
                    if ok:
                        st.success(
                            f"✅ Compte créé avec succès ! "
                            f"Connectez-vous avec **{username}**."
                        )
                        st.session_state["auth_tab"] = "login"
                        import time; time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")

        # ── FOOTER ───────────────────────────────────────────────
        st.markdown("""
        <div class="auth-footer">
          Données collectées sur Instagram, TikTok &amp; YouTube · Tunisie<br>
          Modèle ML : Random Forest · Sentiments : NLP arabe/français<br>
          <span style="color:#cc8db3">© 2025 Salmverse</span>
        </div>
        """, unsafe_allow_html=True)