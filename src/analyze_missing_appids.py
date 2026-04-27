# TODO: refactor AI missing appids
import csv
import re
import os

def extract_appid_from_url(url):
    match = re.search(r'/app/(\d+)/', str(url))
    return match.group(1) if match else None

def is_empty(value):
    if value is None:
        return True
    val_str = str(value).strip()
    return val_str == '' or val_str.lower() == 'nan' or val_str.lower() == 'none'

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    csv_path = os.path.join(base_dir, 'datasets', 'raw', 'steam_games_requirements.csv')
    output_path = os.path.join(base_dir, 'datasets', 'missing_appids_preview.csv')
    
    print(f"Reading: {csv_path}")
    
    missing_appids = []
    total_apps = 0
    total_missing = 0
    
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row.get('url', '')
            app_type = row.get('types', '')
            appid = extract_appid_from_url(url)
            
            if not appid or app_type != 'app':
                continue
            
            total_apps += 1
            
            min_req = row.get('minimum_requirements', '')
            rec_req = row.get('recommended_requirements', '')
            
            missing_min = is_empty(min_req)
            missing_rec = is_empty(rec_req)
            
            if missing_min or missing_rec:
                total_missing += 1
                name = row.get('name', 'Unknown')
                missing_appids.append({
                    'appid': appid,
                    'name': name,
                    'missing_min': 'Yes' if missing_min else 'No',
                    'missing_rec': 'Yes' if missing_rec else 'No'
                })
    
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total apps (type=app) analyzed: {total_apps}")
    print(f"Apps missing requirements:      {total_missing}")
    print(f"Apps with complete requirements: {total_apps - total_missing}")
    print(f"\nSaving to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['appid', 'name', 'missing_min', 'missing_rec'])
        writer.writeheader()
        writer.writerows(missing_appids)
    
    print(f"Successfully saved {len(missing_appids)} appids")
    
    print(f"\n{'='*70}")
    print(f"PREVIEW (first 50 appids)")
    print(f"{'='*70}")
    print(f"{'appid':<10} | {'name':<45} | {'min':<5} | {'rec':<5}")
    print(f"{'-'*70}")
    
    for item in missing_appids[:50]:
        name = item['name'][:42] + '...' if len(item['name']) > 45 else item['name']
        print(f"{item['appid']:<10} | {name:<45} | {item['missing_min']:<5} | {item['missing_rec']:<5}")
    
    if len(missing_appids) > 50:
        print(f"\n... and {len(missing_appids) - 50} more")

if __name__ == '__main__':
    main()