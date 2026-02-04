import pandas as pd
import argparse
import os
from logger_config import logger
import json

def parse_args():
    """
    Parse the arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--tagged_file', type=str,
                        help='Path to the geo-tagged csv files')
    parser.add_argument('--column', type=str,
                        help='Column for which distribution has to be calculated.')
    parser.add_argument('--noun', type=str,
                        help='Noun for which you want distribution for')
    parser.add_argument('--svm', action='store_true',
                    help='enter False if you want distributions for the whole dataset, True if you want for positively classified images')
    parser.add_argument('--copy', action='store_true',
                        help='Want to copy values from one column to another column?')
    parser.add_argument('--source', type=str, default=None,
                        help='Source column to copy from')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name')
    parser.add_argument('--language', type=str, default=None,
                        help='Gre, Spa, Hin, Jap')
    parser.add_argument('--svm_column', type=str, default='svm_1',
                        help='Column name which stores if the image is relevant or irrelevant')
    args = parser.parse_args()
    return args


def get_distribution(df, col, noun, dataset, language):
    """
    Get the country-wise and continent-wise distribution as per llama's outputs for a noun.
    """
    
    distribution = df.groupby([col]).size().reset_index(name='counts')
    distribution = distribution.sort_values(by="counts", ascending=False)
    no_percentage = distribution[distribution[col]=='no']['counts'].sum()/distribution['counts'].sum()
    top_10_percentage = distribution[distribution[col]!='no']['counts'].head(10).sum()/distribution['counts'].sum()
    logger.info(f"Percentage of no: {no_percentage}")
    logger.info(f"Percentage of top 10: {top_10_percentage}")
    logger.info(f"Percentage of others: {1 - no_percentage - top_10_percentage}")
    country_path = f'data/distribution_{noun}_{col}_{dataset}.csv'
    if language is not None:
        country_path = f'data/distribution_{noun}_{col}_{dataset}_{language}.csv'
    distribution.to_csv(country_path, index=False)
    continents = pd.read_csv('data/Countries-Continents.csv')
    continents['Continent'] = continents['Continent'].str.lower()
    continents['Country'] = continents['Country'].str.lower()
    mapping = continents.set_index('Country')['Continent'].to_dict()
    mapping['no'] = 'no'
    df['continent'] = df[col].map(mapping)
    distribution_continent = df.groupby(['continent']).size().reset_index(name='counts_continents')
    distribution_continent = distribution_continent[distribution_continent['continent']!='no']
    distribution_continent['counts_continents'] /= distribution_continent['counts_continents'].sum()
    continent_path = f'data/distribution_{noun}_{col}_{dataset}_continent.csv'
    if language is not None:
        continent_path = f'data/distribution_{noun}_{col}_{dataset}_{language}_continent.csv'
    distribution_continent.to_csv(continent_path, index=False)
    logger.info(distribution_continent)
    return distribution

country_normalizations = json.load(open('country_normalizations.json'))
    

def normalize_country_name(country_name):
    if pd.isna(country_name):
        return 'no'
    if type(country_name) == float:
        return 'no'
    country_lower = str(country_name).strip().lower()
    
    if country_lower == 'czech republic (czechia)':
        return 'czech republic (czechia)'
    country_lower = country_lower.replace('(', '')
    country_lower = country_lower.replace(')', '')
    country_lower = country_lower.replace('\"', '')
    country_lower = country_lower.replace('\'', '')
    country_lower = country_lower.split(':')[-1]
    country_lower = country_lower.strip()
    country_lower = country_lower.replace('_', ' ')
    country_lower = country_lower.split('/')[0]
    country_lower = country_lower.replace('-', ' ')
    country_lower = country_lower.replace('&', 'and')
    
    if country_lower in country_normalizations:
        return country_normalizations[country_lower]
    country_no_the = country_lower.replace('the ', '').strip()
    if country_no_the in country_normalizations:
        return country_normalizations[country_no_the]
    return country_no_the

df_countries = pd.read_csv('data/gdp_population_2025.csv')
# print(df_countries.columns)
all_countries = list(df_countries['Country'])
all_countries = [x.lower() for x in all_countries]
all_countries = [normalize_country_name(x) for x in all_countries] + ['no']

def check_country(text):
    if text not in all_countries:
        # print('no', text)
        return 'NULL'
    return text
    

def find_distribution(args=None):
    df = pd.read_csv(args.tagged_file, engine="python", on_bad_lines='skip')
    df.reset_index(drop=True, inplace=True)
    
    if args.svm:
        num_nan = df['url'].isna().sum()
        print(f"Number of nan values in svm_1: {num_nan}, length of df: {len(df)}")
        df = df[df['svm_1']==1]
        
    if args.source:
        df[args.source] = df[args.source].str.lower()
    df[args.column] = df[args.column].str.lower()
    df[args.column] = df[args.column].apply(normalize_country_name)
    
    df[args.column] = df[args.column].apply(check_country)
    df = df[df[args.column]!='NULL']
    if args.copy:
        for idx, row in df.iterrows():
            if len(row[args.source].split('_')) <= 1:
                df.at[idx, args.column] = df.at[idx, args.source]
    df = get_distribution(df, args.column, args.noun, args.dataset, args.language)
    

if __name__ == '__main__':
    args = parse_args()
    logger.info(f"________________{args.noun}_________________")
    find_distribution(args)
    # print(sorted(all_countries))vene