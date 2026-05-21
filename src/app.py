import pickle
import random
import re
import tomllib
import streamlit as st
import pandas as pd
import numpy as np
from thefuzz import process
from pathlib import Path
import smtplib
from email.message import EmailMessage
import requests

# ── Supabase REST (no WebSocket) ─────────────────────────
@st.cache_resource
def _supabase_config() -> dict:
    try:
        return st.secrets["supabase"]
    except Exception:
        env_path = Path(__file__).parent / ".env"
        with open(env_path, "rb") as f:
            return tomllib.load(f)["supabase"]

def _rest_headers():
    key = _supabase_config()["key"]
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

def email_exists(email: str) -> bool:
    url = f"{_supabase_config()['url'].rstrip('/')}/rest/v1/feedback"
    r = requests.get(
        url,
        params={"email": f"eq.{email}", "select": "email"},
        headers=_rest_headers(),
    )
    r.raise_for_status()
    return len(r.json()) > 0

def save_feedback(email, game, score, cluster, neighbors, rating, observations=""):
    url = f"{_supabase_config()['url'].rstrip('/')}/rest/v1/feedback"
    r = requests.post(
        url,
        json={
            "email": email,
            "game": game,
            "cpu_score": round(score, 1),
            "cluster": cluster,
            "neighbor_1": neighbors[0] if len(neighbors) > 0 else "",
            "neighbor_2": neighbors[1] if len(neighbors) > 1 else "",
            "neighbor_3": neighbors[2] if len(neighbors) > 2 else "",
            "rating": rating,
            "observations": observations,
        },
        headers=_rest_headers(),
    )
    if r.status_code == 409:
        st.warning("Você já enviou feedback com este email.")
        return False
    r.raise_for_status()
    return True

def validate_email(email: str) -> bool:
    return bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email))

def send_verification_code(recipient: str, code: str) -> bool:
    try:
        cfg = st.secrets["smtp"]
    except Exception:
        env_path = Path(__file__).parent / ".env"
        with open(env_path, "rb") as f:
            cfg = tomllib.load(f)["smtp"]
    msg = EmailMessage()
    msg.set_content(f"Seu código de verificação do HardwareMatch é: {code}")
    msg["Subject"] = "HardwareMatch — Código de Verificação"
    msg["From"] = cfg["email"]
    msg["To"] = recipient
    with smtplib.SMTP(cfg["server"], int(cfg["port"])) as server:
        server.starttls()
        server.login(cfg["email"], cfg["password"])
        server.send_message(msg)
    return True

HERE = Path(__file__).parent.resolve()
ARTIFACTS = HERE / "artifacts"

@st.cache_resource
def load_artifacts():
    games = pd.read_pickle(ARTIFACTS / "games.pkl")
    games_filtered = pd.read_pickle(ARTIFACTS / "games_filtered.pkl")
    with open(ARTIFACTS / "knn.pkl", "rb") as f:
        knn = pickle.load(f)
    with open(ARTIFACTS / "cluster_names.pkl", "rb") as f:
        cluster_names = pickle.load(f)
    return games, games_filtered, knn, cluster_names

def score_game_by_name(game, df):
    match = df[df['name'] == game]
    if not match.empty:
        row = match.iloc[0]
        return row['cpu_bench'], row['min_cpu'], row['name'], True
    extracted, accuracy, index = process.extractOne(game, df['name'])
    if accuracy < 80:
        return None, None, None, False
    row = df.iloc[index]
    return row['cpu_bench'], row['min_cpu'], row['name'], True

games, games_filtered, knn, cluster_names = load_artifacts()

st.set_page_config(page_title="HardwareMatch", layout="centered")
st.title("HardwareMatch")
st.markdown("Descubra qual nível de CPU seu jogo exige e encontre jogos com demandas similares.")

# ── Email verification gate ──────────────────────────────
if not st.session_state.get("email_verified"):
    with st.container(border=True):
        st.subheader("📧 Verificação de Email")
        st.markdown("Insira seu email para continuar. Apenas **um feedback por email**.")

        email = st.text_input("Email", placeholder="seu@email.com", key="gate_email")

        if st.button("Enviar código", key="send_code"):
            if not email:
                st.warning("Digite um email.")
            elif not validate_email(email):
                st.error("Formato de email inválido.")
            elif email_exists(email):
                st.error("Este email já enviou um feedback. Apenas um feedback por email é permitido.")
            else:
                code = random.randint(100000, 999999)
                st.session_state.verification_code = str(code)
                st.session_state.pending_email = email
                try:
                    send_verification_code(email, str(code))
                    st.success("Código enviado para seu email!")
                except Exception as e:
                    st.warning(f"Não foi possível enviar o email ({e}). Use o código abaixo:")
                    st.info(f"Código de verificação: **{code}**")
                st.rerun()

        if st.session_state.get("verification_code"):
            code_input = st.text_input("Código de verificação", placeholder="000000", key="gate_code")
            if st.button("Verificar", key="verify_code"):
                if code_input == st.session_state.verification_code:
                    st.session_state.email_verified = True
                    st.session_state.verified_email = st.session_state.pending_email
                    st.success("Email verificado com sucesso!")
                    st.rerun()
                else:
                    st.error("Código incorreto. Tente novamente.")

    st.stop()
# ─────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["🔍 Buscar por Jogo", "📊 Buscar por Pontuação"])

with tab1:
    if st.session_state.pop("feedback_sent", False):
        st.toast("Obrigado pela avaliação!", icon="✅")

    with st.form("search_form"):
        game_input = st.text_input("Nome do jogo", placeholder="Ex: Cyberpunk 2077")
        submitted = st.form_submit_button("Recomendar")

    if submitted and game_input:
        score, cpu_str, matched_name, found = score_game_by_name(game_input, games)
        if score is not None and (isinstance(score, float) and np.isnan(score)):
            st.error(f"Não foi possível determinar a pontuação de CPU para \"{game_input}\".")
            found = False
        if not found:
            st.error(f"Não encontrado: \"{game_input}\".")
        else:
            match_row = games_filtered[games_filtered['cpu_bench'] == score]
            if not match_row.empty:
                cluster = match_row['cluster'].iloc[0]
                label = cluster_names.get(cluster, "Desconhecido")
            else:
                cluster = None
                label = "Abaixo do limite"
            if score is not None:
                distances, indices = knn.kneighbors([[score]], return_distance=True)
                st.session_state.results = dict(
                    score=score, cpu_str=cpu_str, matched_name=matched_name,
                    label=label, game_input=game_input, indices=indices
                )

    if st.session_state.get("results"):
        r = st.session_state.results
        st.subheader(r["matched_name"])
        cols = st.columns(2)
        cols[0].metric("CPU Necessária", r["cpu_str"] or "N/A")
        cols[1].metric("Pontuação do CPU", f"{r['score']:.1f}" if r["score"] else "N/A")
        st.markdown(f"**Nível de Performance:** {r['label']}")

        if r["score"] is not None:
            st.divider()
            st.subheader("Jogos com Requisitos Similares")
            distances, indices = knn.kneighbors([[r["score"]]], return_distance=True)
            neighbors = games_filtered.iloc[indices[0]].copy()
            neighbors['distance'] = distances[0]
            st.dataframe(
                games.loc[neighbors.index][['name', 'min_cpu', 'cpu_bench']],
                width='stretch'
            )
            if not st.session_state.get("feedback_submitted"):
                with st.form("feedback_form"):
                    rating = st.slider("Qual a probabilidade de você comprar essa CPU?", 0, 10, 5)
                    observations = st.text_area("Observações (opcional)", placeholder="Compartilhe sua opinião sobre a recomendação...")
                    if st.form_submit_button("Enviar"):
                        ok = save_feedback(
                            st.session_state.verified_email,
                            r["game_input"], r["score"], r["label"],
                            [games.loc[games_filtered.iloc[idx].name]['name']
                             for idx in r["indices"][0]],
                            rating, observations=observations
                        )
                        if ok:
                            st.session_state.feedback_submitted = True
                            st.session_state.feedback_sent = True
                            del st.session_state.results
                            st.rerun()
            else:
                st.info("Você já enviou seu feedback. Obrigado!")

with tab2:
    score_input = st.number_input("Pontuação do CPU", min_value=0.0, max_value=200.0, step=1.0, value=70.0)
    if st.button("Encontrar Similares", key="score_btn"):
        distances, indices = knn.kneighbors([[score_input]], return_distance=True)
        st.subheader(f"Jogos mais próximos da pontuação {score_input:.1f}")
        st.dataframe(
            games.loc[games_filtered.iloc[indices[0]].index][['name', 'min_cpu', 'cpu_bench']],
            width='stretch'
        )

with st.sidebar:
    st.header("Sobre")
    st.markdown(
        "**HardwareMatch** usa requisitos de jogos da Steam comparados "
        "com pontuações de CPU do UserBenchmark para recomendar jogos "
        "com demandas de CPU similares.\n\n"
        f"Base de dados: {len(games)} jogos com benchmarks de CPU"
    )
    st.divider()
    st.markdown("**Legenda dos Clusters**")
    for cid in sorted(cluster_names.keys()):
        st.markdown(f"- **{cluster_names[cid]}**")
