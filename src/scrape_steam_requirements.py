# TODO: refactor AI code
import pandas as pd
import requests
import time
import csv
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE_DIR / 'datasets' / 'scraped' / 'mostwishlisted_games.csv'
OUTPUT_CSV = BASE_DIR / 'datasets' / 'scraped' / 'steam_requirements_scraped.csv'
INTERVAL = 200

API_URL = "https://store.steampowered.com/api/appdetails?appids={appid}"

def load_missing_appids():
    logger.info(f"Loading missing appids from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    return df['appid'].astype(str).tolist()

def get_existing_scraped():
    if OUTPUT_CSV.exists():
        df = pd.read_csv(OUTPUT_CSV)
        return set(df['steam_appid'].astype(str).tolist())
    return set()

def parse_pc_requirements(pc_req):
    if pc_req is None:
        return '', ''
    if isinstance(pc_req, list):
        return '', ''
    if isinstance(pc_req, dict):
        return pc_req.get('minimum', '') or '', pc_req.get('recommended', '') or ''
    return '', ''

def scrape_appid(appid, retries=3):
    url = API_URL.format(appid=appid)
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if appid in data:
                    app_data = data[appid]
                    if app_data.get('success'):
                        inner = app_data.get('data', {})
                        min_req, rec_req = parse_pc_requirements(inner.get('pc_requirements'))
                        return {
                            'steam_appid': appid,
                            'name': inner.get('name', ''),
                            'pc_requirements_minimum': min_req,
                            'pc_requirements_recommended': rec_req
                        }
                    else:
                        logger.info(f"Unsuccesful: {appid}")
            return None
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None

def save_results(results, mode='a'):
    if not results:
        return
    file_exists = OUTPUT_CSV.exists() and OUTPUT_CSV.stat().st_size > 0
    mode_write = 'w' if mode == 'w' or not file_exists else 'a'
    header = not file_exists
    
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, mode=mode_write, index=False, header=header, encoding='utf-8', quoting=csv.QUOTE_ALL)
    logger.info(f"Saved {len(results)} results. Total file size: {OUTPUT_CSV.stat().st_size} bytes")

def main():

    appids = load_missing_appids()
    logger.info(f"Total appids to scrape: {len(appids)}")
    
    existing = get_existing_scraped()
    logger.info(f"Already scraped: {len(existing)}")
    
    appids_to_scrape = [a for a in appids if a not in existing]
    logger.info(f"To scrape (head=10): {appids_to_scrape[:10]}")
    logger.info(f"Remaining to scrape: {len(appids_to_scrape)}")
    
    if not appids_to_scrape:
        logger.info("Nothing to scrape. Shutting down...")
        return
    
    results = []
    start_time = time.time()
    total_scraped = len(results)
    
    try:
        for idx, appid in enumerate(appids_to_scrape):
            result = scrape_appid(appid)
            if result:
                results.append(result)
                total_scraped += 1
            
            if (idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                remaining = len(appids_to_scrape) - (idx + 1)
                eta = remaining / rate if rate > 0 else 0
                logger.info(f"Progress: {idx + 1}/{len(appids_to_scrape)} | Rate: {rate:.1f} req/s | ETA: {eta/60:.1f} min")
            
            if (idx + 1) % INTERVAL == 0:
                save_results(results, mode='a')
                logger.info(f"Checkpoint saved at {idx + 1}")
                results = []
            
            time.sleep(1.5)
        
        if results:
            save_results(results, mode='a')
        
    except KeyboardInterrupt:
        logger.warning("Interrupted. Saving progress...")
        if results:
            save_results(results, mode='a')
        return
    
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("SCRAPING COMPLETE")
    logger.info(f"Total scraped: {total_scraped}")
    logger.info(f"Time elapsed: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    logger.info(f"Output: {OUTPUT_CSV}")
    logger.info("=" * 60)

if __name__ == '__main__':
    main()