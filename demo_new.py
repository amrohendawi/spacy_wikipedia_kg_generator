# source article https://towardsdatascience.com/auto-generated-knowledge-graphs-92ca99a81121
import os
import pandas as pd

from web_scraper_new import wiki_scrape, wiki_page, get_entity_pairs
from kg_visualization import draw_kg, filter_graph


def main():
    target = 'Grayhound'
    if not os.path.exists(f'data/wiki_data_{target}.csv'):
        wiki_data = wiki_scrape(target)
        wiki_data.to_csv(f'data/wiki_data_{target}.csv', index=False)

    wiki_data = pd.read_csv(f'data/wiki_data_{target}.csv')
    if not os.path.exists(f'data/pairs_{target}.csv'):
        pairs = get_entity_pairs(wiki_data.loc[0, 'text'])
        pairs.to_csv(f'data/pairs_{target}.csv', index=False)

    pairs = pd.read_csv(f'data/pairs_{target}.csv')
    draw_kg(pairs)
    filter_graph(pairs, 'Greyhounds')


if __name__ == '__main__':
    main()
