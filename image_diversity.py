import argparse
import gc
import os
import random
from vendi_score import image_utils
from logger_config import logger
import clip
import joblib
# import lpips
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pylab import rcParams
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr
from sklearn import metrics
from sklearn.metrics.pairwise import pairwise_distances
from torchvision.io import ImageReadMode, read_image
from torchvision.models import ResNet50_Weights
from torchvision.utils import make_grid
from transformers import AutoImageProcessor, AutoModel, BeitModel, DeiTModel
from logger_config import logger
import json

gc.collect()
random.seed = 1234
np.random.seed = 1234

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

def parse_args():
    """
    Parse the arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--noun', type=str, default='house',
                        help='house, car')
    parser.add_argument('--csv-path', type=str, default='None',
                        help='Path to the file')
    parser.add_argument('--img_directory', type=str, default='None',
                        help='Directory')
    parser.add_argument('--dataset', type=str, default='None',
                        help='dataset') 
    parser.add_argument('--country-col', type=str, default='country',
                        help='name of the column containing country name')
    parser.add_argument('--annotation-col', type=str, default='manual_annotations',
                        help='name of the column containing house annotations')
    parser.add_argument('--vendi', action='store_true', default=False,
                        help='do we want to use vendi score?')
    parser.add_argument('--sd', action='store_true', default=False,
                        help='generated images?')
    args = parser.parse_args()
    return args


def extract_img_feats(df, directory):
    """
    CLIP features extracted and stored
    Args:
        df (pandas dataframe): Annotated dataset of the given noun
    Returns:
        ndarray: all images converted to features
        list: indices of images in the input dataframe whose images 
              are considered
    """
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    
    # Load the DiNO model
    model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
    preprocess = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        
    overall_feats = []
    indices = []
    for index, row in df.iterrows():

        id = str(row['id']) #.split('.')[0]
        print(id, 'hellloooo')
        if 'local_path' in df.columns:
            local_path = row['local_path']
        else:
            local_path = os.path.join(directory, f'{id}.jpg')
                
        try:
            inputs = preprocess(Image.open(local_path), return_tensors="pt").to(device)
        except Exception as e:
            print(str(e))
            continue
        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            features = torch.mean(last_hidden_states, dim=1)
                
        indices.append(index)
        features_flat = features.flatten().cpu().numpy()
        features_flat = features_flat / np.linalg.norm(features_flat)
        overall_feats.append(features_flat)

    overall_feats = np.array(overall_feats)
    return overall_feats, indices

def get_images(df=None, directory=None):
    """
    Get the images from the dataframe
    Args:
        df (pandas dataframe): Annotated dataset of the given noun
    Returns:
        list: list of images
    """
    images = []
    if df is None:
        print('Getting images from directory', directory)
        filenames = os.listdir(directory)[:250]
        for filename in filenames:
            if filename.endswith('.jpg') or filename.endswith('.png'):
                images.append(Image.open(os.path.join(directory, filename)).convert('RGB'))
        return images

    for idx, row in df.iterrows():
        id = str(row['id'].lstrip('0')).split('.')[0]
        if 'local_path' in df.columns:
            local_path = row['local_path']
        else:
            local_path = os.path.join(directory, f'{id}.jpg')
        image = Image.open(local_path)
        image = image.convert('RGB')
        images.append(image)
    return images


def compute_distance(overall_feats):
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
    
    mean_vector = np.mean(overall_feats, axis=0).reshape(1, -1)
    std = 0

    for i in range(overall_feats.shape[0]):
        
        min_l2_ind = []
        std += (np.linalg.norm(overall_feats[i].reshape(1, -1) - mean_vector)) ** 2
        
    if overall_feats.shape[0]-1 <= 0:
        std = -1
    else:
        std = np.sqrt((1/overall_feats.shape[0])*std)
    print(std)
    return std

def vendi_fixed_size(images, k, n_trials=10, device='cuda'):
    scores = []
    for idx in range(n_trials):
        print('Sampling images', idx)
        subset = random.sample(images, k)
        print('Subset sampled', len(subset))
        score = image_utils.embedding_vendi_score(subset, device=device)
        scores.append(score)
    return float(np.mean(scores))


if __name__ == '__main__':
    args = parse_args()
    csv_path = args.csv_path
    df = pd.read_csv(csv_path, engine='python', on_bad_lines='skip')
    if 'id' not in df.columns:
        idx = [str(i) for i in range(len(df))]
        df['id'] = idx
    df[args.country_col] = df[args.country_col].str.lower()
    
    df_countries = pd.read_csv('data/gdp_population_2025.csv')
    print(df_countries.columns)
    all_countries = list(df_countries['Country'])
    all_countries = [x.lower() for x in all_countries]
    all_countries = [normalize_country_name(x) for x in all_countries]
    
    print('Total Num of countries', len(all_countries))
    dist_country_dic = {}
    nums = {}
    
    df[args.country_col] = df[args.country_col].str.lower()
    df[args.country_col] = df[args.country_col].apply(normalize_country_name)
    df[args.country_col] = df[args.country_col].apply(check_country)
    df = df[df[args.country_col]!='NULL']
    lens = [len(df[(df[args.country_col] == country) & (df[args.annotation_col] == 1)]) for country in all_countries]
    lens = [x for x in lens if x >= 100]
    min_val = min(lens)
    print('this is the min length', min_val)
    for country in all_countries:
        df1 = df[(df[args.country_col] == country) & (df[args.annotation_col] == 1)]
        df1.reset_index(drop=True, inplace=True)
        
        if len(df1) < min_val:
           continue
        if len(df1) > 50000:
            df1 = df1.sample(50000, replace=False)
        # print(country, len(df1))
        
        ids = df1['id']
        if not args.vendi:
            overall_feats, indices = extract_img_feats(df1, args.img_directory)
            dist1 = compute_distance(overall_feats) 
        else:
            if not args.sd:
                images = get_images(df=df1, directory=args.img_directory)
            else:
                images = get_images(directory=os.path.join(args.img_directory, f'{args.noun}_{country}'))
            print('Images read successfully')
            dist1 = image_utils.embedding_vendi_score(images, device='cuda')
                
        dist_country_dic[country] = dist1
        
        nums[country] = len(images)
        
        print('noun: ', args.noun, 'country done', country)
    all_countries_sorted = list(dist_country_dic.keys())
    all_countries_sorted.sort()
    sorted_dist = {i: dist_country_dic[i] for i in all_countries_sorted}
    
    sorted_num = {i: nums[i] for i in all_countries_sorted}
    
    df_diversity = pd.DataFrame()
    df_diversity['Countries'] = list(sorted_dist.keys())
    df_diversity['Frequencies'] = list(sorted_num.values())
    df_diversity['std'] = list(sorted_dist.values())
    if not args.sd:
        df_diversity.to_csv(f'data/Diversity_{args.noun}_{args.dataset}.csv', index=False)
    else:
        df_diversity.to_csv(f'data/Diversity_{args.noun}_{args.dataset}_sd.csv', index=False)
    logger.info(f"Noun: {args.noun}")
    logger.info(f"Frequency vs Image Corr {spearmanr(df_diversity['Frequencies'], df_diversity['std'])}")
    mean_value = df_diversity['std'].mean()
    std_value = df_diversity['std'].std()
    logger.info(f"Image Div Mean and Std {mean_value}, {std_value}")