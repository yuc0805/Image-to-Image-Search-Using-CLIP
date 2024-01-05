"""

Kritik Seth, "Fruits and Vegetables Image Recognition Dataset," Kaggle 2020 
[https://www.kaggle.com/kritikseth/fruit-and-vegetable-image-recognition]

"""

import argparse
import numpy as np
import os
import sys
import PIL

import torch
from tqdm import tqdm
import glob

from utils import get_model, plotSimilarImages

def get_args_parser():
    parser = argparse.ArgumentParser('Image-to-Image Search Engine',add_help=False)
    
    parser.add_argument('--mode', default='train', type=str,
                        help='train mode or eval mode')
    
    # Dataset parameters
    parser.add_argument('--data_path',default='./Fruit-Vegetables-Images', type=str,
                        help='dataset path')
    parser.add_argument('--test_set',default='display',type=str,
                        help='the folder name of test_set')
    # Model parameters
    parser.add_argument('--topk', default=7, type=int,
                        help='return top k similar imgaes')
    parser.add_argument('--model_name', default='clip', type=str,
                        help='The model name use to extract feature, recommend:[clip,resnet34,resnet50,]')
    parser.add_argument('--clip_base', default='ViT-B/32', type=str,
                        help="Model base for CLIP, available models = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']")
    parser.add_argument('--input_size', default=124, type=int,
                        help='image input size')

    # Training parameters
    parser.add_argument('--device', default='mps',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    # Saving
    parser.add_argument('--save_dir',default='./output_dir',
                        help='path for saving')
    parser.add_argument('--feature_dict_file',default='corpus_feature_dict.npy',
                        help='filename for saving images features')

    return parser



def main(args):

    #fix seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    model, preprocess = get_model(args.model_name,args)

    if args.mode == 'train':
        print(f'using pretrained model {args.model_name} to extract features')

        all_features = {}
        root = os.path.join(args.data_path,'train','*','*.jpg')
        print(f'current directory: {root}')

        for image_file in tqdm(glob.glob(root)):
            features = get_features(args,image_file,model=model,preprocess=preprocess)
            all_features[image_file] = features

        os.makedirs(f"{args.save_dir}/{args.model_name}",exist_ok=True)
        np.save(f"{args.save_dir}/{args.model_name}/{args.feature_dict_file}",all_features)

        print(f'finish training set feature extraction, and save it into {args.save_dir}/{args.model_name}/{args.feature_dict_file}')

    else:
        print(f'using pretrained model {args.model_name} to search {args.topk} similar images')

        # load feature vectors for pre_train images
        feature_path = f"{args.save_dir}/{args.model_name}/{args.feature_dict_file}"
        if os.path.exists(feature_path):
            all_features = np.load(feature_path,allow_pickle=True)
            all_features = all_features.item()
        else:
            print("Do not have feature extraction for this model_based, please set mode=train to extract feature first")
            sys.exit()

        # read test image and calculate its feature vector
        test_image = glob.glob(os.path.join(args.data_path,args.test_set,'*','*.jpg'))
        test_image += glob.glob(os.path.join(args.data_path,args.test_set,'*','*.png'))
        test_image += glob.glob(os.path.join(args.data_path,args.test_set,'*','*.jpeg'))

        for image_file in tqdm((test_image)):
            features = get_features(args,image_file,model=model,preprocess=preprocess)
            all_features[image_file] = features
            
        sim, keys = Cosine_Similarity_Matrix(all_features)
        result = {}

        for image_file in tqdm(test_image):
            print(f"Sorting most similar images as {image_file}...")
            index = keys.index(image_file)
            sim_vec = sim[index]
            indexs = np.argsort(sim_vec)[::-1][1:args.topk]
            simImages, simScores = [], []

            for idx in indexs:
                simImages.append(keys[idx])
                simScores.append(sim_vec[idx])

            result[image_file] = (simImages,simScores)

        print("displaying similar images ...")
        for image_file in test_image:
            plotSimilarImages(args,image_file,result[image_file][0],result[image_file][1],numRow=1,numCol=args.topk)

        print('Done!')


def get_features(args,image_file,model,preprocess):
        image = PIL.Image.open(image_file)
        image_input = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            if args.model_name == 'clip':
                image_input = image_input.to(args.device)
                model = model.to(args.device)
                image_features = model.encode_image(image_input)  
            else:
                features = model.forward_features(image_input)
                image_features = model.global_pool(features)

        image_features = image_features.squeeze().cpu().numpy()
        return image_features


def Cosine_Similarity_Matrix(feature_dict):
    v = np.array(list(feature_dict.values()))
    
    numerator = np.matmul(v,v.T)
    a = np.linalg.norm(v,axis=1,keepdims=True)
    denominator = np.matmul(a,a.T)

    sim = numerator/denominator
    keys = list(feature_dict.keys())

    return sim,keys




if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)