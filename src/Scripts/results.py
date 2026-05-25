from pathlib import Path
import tomllib, requests
import pandas as pd

def _supabase_config() -> dict:
    env_path = Path(__file__).parent.parent / ".env"
    with open(env_path, "rb") as f:
        return tomllib.load(f)["supabase"]

def _rest_headers():
    key = _supabase_config()["key"]
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


url = f"{_supabase_config()['url'].rstrip('/')}/rest/v1/feedback"
r = requests.get(
    url,
    params={"select": "id, cpu_score, cluster, neighbor_1, neighbor_2, neighbor_3, rating, observations, game_input, game"},
    headers=_rest_headers(),
)
data = r.json()
df = pd.DataFrame(data)
df.to_csv(Path(__file__).parent.parent.parent / 'datasets' / 'results' / 'results.csv', index=False)