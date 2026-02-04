import joblib
import pandas as pd
from openpyxl import load_workbook, Workbook
import os
from PIL import Image
# import lpips
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer
import numpy as np
import random
from vendi_score import text_utils
import clip
from sklearn import metrics
import torch
from scipy.spatial.distance import cdist
import torchvision.models as models
from torchvision.utils import make_grid
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from pylab import rcParams
from torchvision.models import ResNet50_Weights
import argparse
from torchvision.io import read_image, ImageReadMode
import gc
from logger_config import logger
import json

gc.collect()
random.seed = 1234
np.random.seed = 1234

def parse_args():
    """
    Parse the arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--noun', type=str, default='house',
                        help='house, car')
    parser.add_argument('--csv-path', type=str, default='None',
                        help='Path to the file')
    parser.add_argument('--dataset', type=str, default='None',
                        help='Dataset')
    parser.add_argument('--country', type=str, default='india',
                        help='india, united states, united kingdom, nigeria,\
                        kenya, australia, brazil, philippines, guinea,\
                        thailand, all')    
    parser.add_argument('--distance_metric', type=str, default='cosine',
                        help='cosine, euclidean') 
    parser.add_argument('--diversity-metric', type=str, default='knn-ratio',
                        help='remote-edge, remote-clique, remote-star, \
                        remote-pseudoforest, knn-ratio')  
    parser.add_argument('--k1', type=int, default=None)
    parser.add_argument('--k2', type=int, default=None) 
    parser.add_argument('--vendi', action='store_true', default=False,
                        help='do we want to use vendi score?')    
    # parser.add_argument('--model', type=str, default='clip',
    #                     help='cnn or clip?')
    # parser.add_argument('--visualize', action='store_true',
    #                     help='Do you want to visualize the images?')  
    parser.add_argument('--country-col', type=str, default='country',
                        help='name of the column containing country name')
    parser.add_argument('--annotation-col', type=str, default='manual_annotations',
                        help='name of the column containing house annotations')
    args = parser.parse_args()
    return args

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

def check_country(text):
    if text not in all_countries:
        # print('no', text)
        return 'NULL'
    return text


def extract_img_feats(df):
    """
    CLIP features extracted and stored
    Args:
        df (pandas dataframe): Annotated dataset of the given noun
    Returns:
        ndarray: all images converted to features
        list: indices of images in the input dataframe whose images 
              are considered
    """
    # print('HEEEEELOOOOOO')
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sentence_model.encode(df['text'], show_progress_bar=True)
    # embeddings = embeddings / np.linalg.norm(embeddings)  
    # print(embeddings.shape[0])
    for i in range(embeddings.shape[0]):
        # print('done')
        embeddings[i] = embeddings[i] / np.linalg.norm(embeddings[i])
    return embeddings


def compute_distance(df, overall_feats, indices, metric, args):
    """
    Compute diversity. For each datapoint, we five different metrics
    specified in the literature.
    Args:
        overall_feats (numpy.ndarray): image features
        indices (list): indices of the images in the dataframe
        metric (str): metric over each the distance should be computed
    Returns:
        (float): metric computed over every data point
    """
    overall_l2 = []
    avg_dist = 0
    star_value = 999
    pseudoforest_value = 0
    knn_ratio_val = 0
    clique_value = 0
    mean_vector = np.mean(overall_feats, axis=0).reshape(1, -1)
    std = 0

    for i in range(overall_feats.shape[0]):
        
        min_l2_ind = []
        std += (np.linalg.norm(overall_feats[i].reshape(1, -1) - mean_vector)) ** 2
    std = np.sqrt((1/overall_feats.shape[0])*std)
    return std


def visualize(df, country, noun):
    images = list(df['id'])
    images = [f'/data2/abhipsa/datasets/laion_{noun}1M_images/{x}.jpg' for x in images]
    images = [image for image in images if image != 'invalid']
    images = [np.array(Image.open(image).resize((256, 256)).convert("RGB")) for image in images]
    
    images = [torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) for image in images]
    
    Grid = make_grid(images, nrow=int(np.ceil(np.sqrt(len(images)))), padding=1)
    img = Image.fromarray(Grid.permute(1, 2, 0).numpy().astype('uint8'))
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'Plots/{country}_images.png')
    print('Done country', country)

if __name__ == '__main__':
    args = parse_args()
    # csv_path = f'data/{args.noun}_geo_balanced.csv'
    csv_path = args.csv_path
    df = pd.read_csv(csv_path, engine='python', on_bad_lines='skip')
    df[args.country_col] = df[args.country_col].str.lower()
    print(df.columns)
    logger.info(f'Noun: {args.noun} Dataset: {args.dataset}')
    # else:
    # all_countries = ['algeria', 'India', 'Japan', 'South Korea', 'Israel', 'Finland', 'United States', 
    #     'United Kingdom', 'Germany', 'France', 'Italy', 'Spain', 'Greece',
    #     'Czech Republic', 'Poland', 'Chile', 'Brazil', 'Mexico', 'Canada',
    #     'Australia', 'New Zealand', 'Portugal', 'Hungaria', 'South Africa', 'Slovenia',
    #     'Estonia', 'Latvia', 'Belgium', 'No', 'Guinea', 'Nigeria', 'Philippines', 'Kenya', 'Thailand',
    #     'Russia', 'Pakistan', 'Argentina', 'Bangadesh', 'China', 'Ukraine', 'afghanistan',
    #     'armenia', 'albania', 'andorra', 'indonesia', 'singapore', 'austria','bahamas',
    #     'bahrain', 'bali', 'belarus', 'netherlands', 'belize', 'bhutan', 'bermuda', 'bolivia',
    #     'Bosnia and Herzegovina', 'Botswana', 'uruguay', 'colombia', 'brunei', 'bulgaria',
    #     'burkina faso', 'myanmar', 'burundi', 'cabo verde', 'cambodia', 'venezuela', 'cameroon',
    #     'chad', 'jamaica', 'laos', 'comoros', 'croatia', 'ecuador', 'cuba', 'cyprus', 'denmark',
    #     'djibouti', 'dominican republic', 'east timor', 'egypt', 'iraq', 'iran', 'yemen', 
    #     'equatorial guinea', 'estonia', 'eswatini', 'ethiopia', 'falkland islands', 'faroe islands',
    #     'fiji', 'qatar', 'turkey', 'french polynesia', 'gabon', 'gambia', 'gaza', 'georgia',
    #     'norway', 'taiwan', 'ghana', 'gibraltar', 'greenland', 'guam', 'guatemala',
    #     'guinea bissau', 'guyana', 'haiti', 'honduras', 'hong kong', 'sri lanka', 'syria',
    #     'malta', 'malawi', 'isle of man', 'jordan', 'kazakhstan', 'kiribati', 'kosovo',
    #     'kuwait', 'kyrgyzstan', 'lebanon', 'lesotho', 'liberia', 'libya', 'liechtenstein',
    #     'lithuania', 'luxembourg', 'macedonia', 'madagascar', 'malawi', 'malaysia', 'maldives',
    #     'marshall islands', 'mauritius', 'mauritania', 'moldova', 'monaco', 'mongolia',
    #     'montenegro', 'morocco', 'mozambique', 'namibia', 'nepal', 'nicaragua', 'niger', 
    #     'niue', 'north korea', 'oman', 'palau', 'papua new guinea', 'paraguay', 'peru',
    #     'mali', 'serbia', 'senegal', 'sierra leone', 'vietnam', 'zambia', 'romania',
    #     'saint kitts and nevis', 'saint lucia', 'saint vincent and the grenadines',
    #     'samoa', 'sao tome and principe', 'saudi arabia', 'seychelles', 'slovakia',
    #     'solomon islands', 'somalia', 'sudan', 'suriname', 'sweden', 'switzerland',
    #     'tajikistan', 'tanzania', 'tonga', 'trinidad and tobago', 'tunisia', 'turkmenistan',
    #     'turks and caicos islands', 'tuvalu', 'uganda', 'united arab emirates',
    #     'uzbekistan', 'vatican city', 'zimbabwe', 'dr congo', 'British Virgin Islands',
    #     'American Samoa', 'Angola', 'Anguilla', 'Aruba', 'Azerbaijan', 'Barbados', 'Benin',
    #     'Cape Verde', 'Cayman Islands', 'Central African Republic', 'Cook Islands',
    #     'costa rica', 'Curacao', 'Dominica', 'El Salvador', 'Eritrea', 'Grenada',
    #     'Ivory Coast', 'Macau', 'Micronesia', 'Montserrat', 'Nauru', 'New Caledonia',
    #     'North Macedonia', 'Northern Mariana Islands', 'Palestine', 'Panama',
    #     'Puerto Rico', 'Republic of the Congo', 'Rwanda', 'Saint Martin', 'San Marino',
    #     'Sint Maarten', 'Vanuatu']
    # all_countries = ['nepal']
    df_countries = pd.read_csv('data/gdp_population_2025.csv')
    print(df_countries.columns)
    all_countries = list(df_countries['Country'])
    all_countries = [x.lower() for x in all_countries]
    all_countries = [normalize_country_name(x) for x in all_countries]
    print('Total Num of countries', len(all_countries))
    dist_country_dic = {}
    dist_country_dic1 = {}
    dist_country_dic2 = {}
    nums = {}
    # df[args.country_col] = df[args.country_col].to_string()
    # df[args.country_col] = df[args.country_col].dropna()
    df[args.country_col] = df[args.country_col].str.lower()
    df[args.country_col] = df[args.country_col].apply(normalize_country_name)
    df[args.country_col] = df[args.country_col].apply(check_country)
    lens = [len(df[(df[args.country_col] == country) & (df[args.annotation_col] == 1)]) for country in all_countries]
    lens = [x for x in lens if x >= 100]
    min_val = min(lens)
    print('this is the min length', min_val)
    for country in all_countries:
        df1 = df[(df[args.country_col] == country) & (df[args.annotation_col] == 1)]
        df1.reset_index(drop=True, inplace=True)
        # print(len(df1))
        # if len(df1) >= min_val:
        #     print('reached')
        #     df1 = df1.sample(n=min_val, random_state=1234)
        #     df1.reset_index(drop=True, inplace=True)
        #     # 
        # else:
        #     continue
        if len(df1) < 100:
            continue
        if len(df1) > 50000:
            df1 = df1.sample(50000, replace=False)
        print(country, len(df1))
        nums[country] = len(df1)
        if not args.vendi:
            overall_feats = extract_img_feats(df1)
            print('reached')
            dist = compute_distance(df1, overall_feats, None, 
                                        args.distance_metric, args)
        else:
            dist =text_utils.embedding_vendi_score(df1['text'].values.tolist(), model_path="princeton-nlp/unsup-simcse-bert-base-uncased", device="cuda")

        dist_country_dic[country] = dist
            # del overall_feats
            # gc.collect()
        print('country done', country)
    all_countries_sorted = list(dist_country_dic.keys())
    all_countries_sorted.sort()
    sorted_dist = {i: dist_country_dic[i] for i in all_countries_sorted}
    
    sorted_num = {i: nums[i] for i in all_countries_sorted}
    # denom = sum([dist_country_dic[c]/nums[c] for c in all_countries])
    # dist_country_dic = {k:(v/denom - 1) for k,v in dist_country_dic.items()}
    
        # rcParams['figure.figsize'] = 10, 10
        # plt.bar(range(len(list(dist_country_dic.keys()))),
        #         list(dist_country_dic.values()))
        # plt.xticks(range(len(list(dist_country_dic.keys()))),
        #             list(dist_country_dic.keys()), rotation=90)
        # plt.xlabel('Countries')
        # # plt.ylim(0, 1)
        # plt.ylabel(f'Diversity Scores ({args.diversity_metric})') 
        # if args.diversity_metric == 'knn-ratio':
        #     plt.savefig(f'Plots/diversity_{args.diversity_metric}_{args.k1}.png')
        # else:
        #     plt.savefig(f'Plots/diversity_{args.diversity_metric}.png')

    # if os.path.exists(f'data/Diversity_{args.noun}.csv'):
    #     df_diversity = pd.read_csv(f'data/Diversity_{args.noun}.csv')
    #     print('Already existed')
    # else:
    
    df_diversity = pd.DataFrame()
    countries = list(sorted_dist.keys())
    frequencies = [sorted_num[i] for i in countries]
    text_std = [sorted_dist[i] for i in countries]
    df_diversity['Countries'] = countries
    df_diversity['Frequencies'] = frequencies
    # df_diversity[args.diversity_metric] = list(sorted_dist.values())
    df_diversity['text_std'] = text_std
    df_diversity.to_csv(f'data/Diversity_{args.noun}_text_{args.dataset}.csv', index=False)
    logger.info(f"Frequency vs Text Corr: {pearsonr(df_diversity['Frequencies'], df_diversity['text_std'])}")
    mean_value = df_diversity['text_std'].mean()
    std_value = df_diversity['text_std'].std()
    logger.info(f"Text Div Mean and Std {mean_value}, {std_value}")
        


