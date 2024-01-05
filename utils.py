import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import PIL
import tqdm

import torch
import torchvision.transforms as transforms

import timm
import clip


def get_model(model_name,args):
    if model_name == 'clip':
        model, preprocess = clip.load(args.clip_base, device=args.device)
    
    # load timm model
    else:
        model = timm.create_model(model_name,pretrained=True)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of trainable parameters (M): %.2f'%(n_parameters/1e6))
        model.eval()
        preprocess = _preprocess(args.input_size)

    return model, preprocess


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _preprocess(n_px):
    transform = transforms.Compose([
                transforms.Resize((n_px,n_px)),
                _convert_image_to_rgb,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet default statistics
            ])
    return transform


def plotSimilarImages(args,image,simImages,simValues,numRow=1,numCol=4):
    fig = plt.figure()

    fig.set_size_inches(15,15)
    fig.suptitle(f'engine model: {args.model_name}',fontsize=35)

    for j in range(0,numCol*numRow):
        ax = []
        if j == 0:
            img = PIL.Image.open(image)
            ax = fig.add_subplot(numRow,numCol,j+1)
            setAxes(ax,image.split(os.sep)[-1],query=True)
        else:
            img = PIL.Image.open(simImages[j-1])
            ax.append(fig.add_subplot(numRow,numCol,j+1))
            setAxes(ax[-1],simImages[j-1].split(os.sep)[-1],value=simValues[j-1])
        img = img.convert('RGB')
        plt.imshow(img)
        img.close()

    figname = f"{args.model_name}_search_top_{args.topk}_{image.split(os.sep)[-1].split('.')[0]}.png"
    figpath = f"{args.save_dir}/{args.model_name}/image"
    os.makedirs(figpath,exist_ok=True)
    fig.savefig(os.path.join(figpath, figname))
    plt.show()
    plt.close(fig)
    
    

def setAxes(ax,image,query=False,**kwargs):
    value = kwargs.get('value',None)
    if query:
        ax.set_xlabel('Query Image\n{0}'.format(image),fontsize=12)
        ax.xaxis.label.set_color('red')
    else:
        ax.set_xlabel('score={1:1.3f}\n{0}'.format(image,value),fontsize=12)
        ax.xaxis.label.set_color('blue')

    ax.set_xticks([])
    ax.set_yticks([])

