from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path

appids = []
game_names = []

folder = Path("../../datasets/html/")
html_files = list(folder.glob("*.html"))
for filepath in html_files:
    with open(filepath, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, 'html.parser')
        tags = soup.find_all(class_="app")
        counter = 0
        for tag in tags:
            if counter >= 7 or counter == 6:
                game_name = tag.find_next(class_='b').text
            else:
                game_name = tag.find_next('span').text     

            appid = tag['data-appid']
            if not appid in appids:
                appids.append(tag['data-appid'])
                game_names.append(game_name)
            
            counter+=1

pd.DataFrame({'appid': appids, 'name': game_names}).to_csv("../../datasets/scraped/steamdb_relevant_games.csv")