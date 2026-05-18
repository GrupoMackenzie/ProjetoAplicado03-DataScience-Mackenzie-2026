from bs4 import BeautifulSoup
import pandas as pd

appids = []
game_names = []

with open("../datasets/steamdb_mostwishlisted_games.html", "r", encoding="utf-8") as file:
    soup = BeautifulSoup(file, 'html.parser')
    tags = soup.find_all(class_="app")
    counter = 0
    for tag in tags:
        if counter >= 7 or counter == 6:
            game_names.append(tag.find_next(class_='b').text)
        else:
            game_names.append(tag.find_next('span').text)

        appids.append(tag['data-appid'])
        counter+=1

pd.DataFrame({'appid': appids, 'name': game_names}).to_csv("../datasets/scraped/mostwishlisted_games.csv")