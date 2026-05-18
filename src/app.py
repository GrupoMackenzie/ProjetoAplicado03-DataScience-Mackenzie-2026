import pickle
import streamlit as st
import pandas as pd
import numpy as np
from thefuzz import process
from pathlib import Path
import csv
from datetime import datetime

FEEDBACK_FILE = Path(__file__).parent / "feedback.csv"
def save_feedback(game, score, cluster, neighbors, rating):
    if not FEEDBACK_FILE.exists():
        with open(FEEDBACK_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "timestamp","game","cpu_score","cluster",
                "neighbor_1","neighbor_2","neighbor_3","rating"
            ])
    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.now().isoformat(), game,
            f"{score:.1f}", cluster, *neighbors, rating
        ])

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
            st.subheader("Jogos Similares")
            distances, indices = knn.kneighbors([[r["score"]]], return_distance=True)
            neighbors = games_filtered.iloc[indices[0]].copy()
            neighbors['distance'] = distances[0]
            st.dataframe(
                games.loc[neighbors.index][['name', 'min_cpu', 'cpu_bench']],
                width='stretch'
            )
            with st.form("feedback_form"):
                rating = st.slider("Qual a probabilidade de você comprar essa CPU?", 0, 10, 5)
                if st.form_submit_button("Enviar"):
                    save_feedback(
                        r["game_input"], r["score"], r["label"],
                        [games.loc[games_filtered.iloc[idx].name]['name']
                         for idx in r["indices"][0]],
                        rating
                    )
                    st.session_state.feedback_sent = True
                    del st.session_state.results
                    st.rerun()

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
