import numpy as np
import torch
import clip
import pandas as pd
import os
import requests
from io import BytesIO
from PIL import Image
import joblib
import time
import argparse
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import argparse
# from logger_config import logger

class CustomDataset(Dataset):
    def __init__(self, df, noun, transform=None, directory=None):
        self.data = df
        self.directory = directory
        self.transform = transform
        self.noun = noun

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # image_path = f'/data2/abhipsa/datasets/laion_{self.noun}1M_images/'+self.data.iloc[idx]['id']+'.jpg'
        image_path = os.path.join(self.directory, self.data.iloc[idx]['id'].split('/')[-1] + '.jpg')
        try:
            image = Image.open(image_path)
            print("Image opened at", idx)
            return preprocess(image), idx
        except Exception:
            print("Skipping image at ", idx)
            return preprocess(Image.new('RGB', (224, 224), (0, 0, 0))), -1

def parse_args():
    """
    Parse the arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--noun', type=str,
                        help='Enter the noun for which you want to classify')
    parser.add_argument('--device', type=str, default='cuda',
                    help='Enter the device: cuda:0, cuda:1, cuda:2, cuda:3')
    parser.add_argument('--csv_path', type=str, help='Enter path of the csv file')
    # parser.add_argument('--dataset', type=str, help='Which dataset do you wish to operate on?')
    parser.add_argument('--directory', type=str, help='Image folder')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    start_time = time.time()
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parse_args()
    noun = args.noun
    device = args.device
    model, preprocess = clip.load("ViT-B/32", device=device)
    model = model.to(device)
    # Create a DataLoader with efficient data loading and batch processing
    batch_size = 2048
    
    csv_path = args.csv_path
    df = pd.read_csv(csv_path, engine='python', on_bad_lines='skip')
    df.reset_index(drop=True, inplace=True)
    '''Uncomment the below lines if the image names are set according to their row number'''
    # idx = [str(i) for i in range(len(df))]
    # df['id'] = idx
    dataset = CustomDataset(df, args.noun, directory=args.directory)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize SVM classifier
    svm_classifier = SVC()


    # Load the trained SVM classifier
    svm_classifier = joblib.load(f'models/svm_classifier_{noun}_1.joblib')

    # Predict labels for images in batches and save the predictions
    predictions = [-1]*len(df)
    for batch, batch_indices in loader:
        images = batch.to(device)  # Move images to GPU if available
        with torch.no_grad():
            # Extract features from images using CLIP
            batch_features = model.encode_image(images)
            # Predict labels using the SVM classifier
            batch_predictions = svm_classifier.predict(batch_features.cpu().numpy())
            # Update predictions for valid images
            for i, idx in enumerate(batch_indices):
                if idx != -1:
                    predictions[idx] = batch_predictions[i]

    # Add predictions to the DataFrame
    df['svm_1'] = predictions
    df.to_csv(csv_path, index=False)
    end_time = time.time()
    print(f"Total Time: {end_time - start_time}")
