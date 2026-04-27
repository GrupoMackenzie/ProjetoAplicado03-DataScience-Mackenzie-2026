# TODO: refactor AI code

import pandas as pd
import requests
import time
import csv
import os
import logging
import subprocess
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE_DIR / 'datasets' / 'missing_appids_preview.csv'
OUTPUT_CSV = BASE_DIR / 'datasets' / 'steam_requirements_scraped.csv'
ERROR_LOG = BASE_DIR / 'src' / 'scrape_errors.log'
INTERVAL = 200

API_URL = "https://store.steampowered.com/api/appdetails?appids={appid}"

def setup_file_handler():
    if ERROR_LOG.exists():
        ERROR_LOG.unlink()
    file_handler = logging.FileHandler(ERROR_LOG, encoding='utf-8')
    file_handler.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

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

def shutdown_pc():
    logger.info("=" * 60)
    logger.info("SHUTTING DOWN PC IN 60 SECONDS...")
    logger.info("=" * 60)
    subprocess.run(['shutdown', '/s', '/t', '60', '/c', 'Steam scraper complete. PC will shutdown in 60 seconds.'])

def main():
    setup_file_handler()
    logger.info("=" * 60)
    logger.info("STEAM REQUIREMENTS SCRAPER")
    logger.info("PC WILL SHUTDOWN WHEN COMPLETE")
    logger.info("=" * 60)
    
    appids = load_missing_appids()
    logger.info(f"Total appids to scrape: {len(appids)}")
    
    existing = get_existing_scraped()
    logger.info(f"Already scraped: {len(existing)}")
    
    appids_to_scrape = [a for a in appids if a not in existing]
    logger.info(f"Remaining to scrape: {len(appids_to_scrape)}")
    
    if not appids_to_scrape:
        logger.info("Nothing to scrape. Shutting down...")
        shutdown_pc()
        return
    
    results = []
    start_time = time.time()
    total_scraped = len(existing)
    
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
        shutdown_pc()
        return
    
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("SCRAPING COMPLETE")
    logger.info(f"Total scraped: {total_scraped}")
    logger.info(f"Time elapsed: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    logger.info(f"Output: {OUTPUT_CSV}")
    logger.info("=" * 60)
    
    shutdown_pc()

if __name__ == '__main__':
    main()