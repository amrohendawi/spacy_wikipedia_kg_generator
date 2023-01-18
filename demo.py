# source article https://towardsdatascience.com/auto-generated-knowledge-graphs-92ca99a81121
import argparse
import os
import pandas as pd

from web_scraper import wiki_scrape
from kg_visualization import draw_kg, filter_graph

from EntityExtractor import EntityExtractor

def main(args):
    entityExtractor = EntityExtractor()

    if not os.path.exists(f'data/wiki_data_{args.target}.csv'):
        wiki_data = wiki_scrape(args.target)
        wiki_data.to_csv(f'data/wiki_data_{args.target}.csv', index=False)

    wiki_data = pd.read_csv(f'data/wiki_data_{args.target}.csv')
    if not os.path.exists(f'data/pairs_{args.target}.csv'):
        pairs = entityExtractor.get_entity_pairs(wiki_data.loc[0, 'text'])
        pairs.to_csv(f'data/pairs_{args.target}.csv', index=False)

    pairs = pd.read_csv(f'data/pairs_{args.target}.csv')
    draw_kg(pairs)
    filter_graph(pairs, args.sub_graph_target)

    # Tests on dialogue summarizations
    # target = "DHGN_test"
    # with open(f'data/{args.target}.csv', 'r') as f:
    #     data = f.read()
    #     pairs = get_entity_pairs(data)
    #     pairs.to_csv(f'data/pairs_{args.target}.csv', index=False)
    #     pairs = pd.read_csv(f'data/pairs_{args.target}.csv')
    #     draw_kg(pairs)
    #     filter_graph(pairs, 'jane')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='Greyhounds', help='target entity')
    parser.add_argument('--sub-graph-target', type=str, default='Greyhounds',
                        help='target entity to create sub-graph from')
    args = parser.parse_args()
    main(args)
