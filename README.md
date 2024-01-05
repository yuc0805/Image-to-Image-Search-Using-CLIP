# Image-to-Image Search Engine

This Python script serves as an image-to-image search engine that allows you to extract features from a dataset using a pretrained model and perform similarity searches between images.

## Model Details
This search engine was specifically designed to experiment with the Contrastive Language-Image Pretraining (CLIP) model, utilizing only its image encoder component. Additionally, the system is adaptable to employ alternative models such as resnet50 and others supported by the timm library for extracting image features. This flexibility allows users to explore a variety of models and their performances within the image-to-image search context.

## Dataset

The code utilizes the "Fruits and Vegetables Image Recognition Dataset" by Kritik Seth, available on Kaggle [here](https://www.kaggle.com/kritikseth/fruit-and-vegetable-image-recognition). Make sure to download and organize your dataset accordingly.

## Dependencies

Ensure you have the necessary dependencies installed. You can do this by running:

```bash
pip install -r requirements.txt
```

## Usage
Clone the repository:
```bash
git clone https://github.com/your-username/Image-to-Image-Search-Using-CLIP.git
cd Image-to-Image-Search-Using-CLIP
```

Install PyTorch 1.7.1 (or later) and torchvision, as well as small additional dependencies, and then install this repo as a Python package. On a CUDA GPU machine, the following will do the trick:

```bash
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
```

### Training Mode

To extract features from a training set and save them:

```bash
python search_main.py --mode=train
```

This command will use a pretrained model (`args.model_name`) to extract features from images in the specified training set (`args.data_path/train`). The features will be saved in the directory specified by `args.save_dir`.

### Evaluation Mode

To search for similar images in a test set:

```bash
python search_main.py --mode=test
```

This command will use a pretrained model (`args.model_name`) to search for the top `k` similar images in the specified test set (`args.data_path/test_set`). The results will be displayed, showing similar images for each test image.

## Configuration

- `--mode`: Specify the mode ('train' or 'test').
- `--data_path`: Path to the dataset.
- `--test_set`: Folder name of the image set to search.
- `--topk`: Number of similar images to retrieve.
- `--model_name`: Model name for feature extraction ('clip', 'resnet34', 'resnet50', etc.).
- `--clip_base`: Model base for CLIP (if using CLIP).
- `--input_size`: Image input size.
- `--device`: Device to use for training/testing.
- `--seed`: Seed for reproducibility.
- `--save_dir`: Path for saving extracted features.
- `--feature_dict_file`: Filename for saving image features.

By default, we have configured the CLIP model for feature extraction. However, you have the flexibility to experiment with various models supported by the timm library for feature extraction. Additionally, if you choose to stick with CLIP, you can modify the clip_base parameter with options like ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']. Feel free to explore and find the model configuration that best suits your needs.

Here's an example how you can train and evaluate a ResNet50 for the below comparision:

```bash
python search_main.py --mode=train --model_name=resnet50
```
```bash
python search_main.py --mode=test --model_name=resnet50
```

## Result

CLIP-based model shows more capable to handle senmatic information in the image, below is a couple example image that shows this:
![picture of banana bread using CLIP](https://github.com/yuc0805/Image-to-Image-Search-Using-CLIP/blob/main/output_dir/clip/image/clip_search_top_7_Banana-Bread2.png)

![picture of banana bread using ResNet50](https://github.com/yuc0805/Image-to-Image-Search-Using-CLIP/blob/main/output_dir/resnet50/image/resnet50_search_top_7_Banana-Bread2.png)

![picture of wine using CLIP](https://github.com/yuc0805/Image-to-Image-Search-Using-CLIP/blob/main/output_dir/clip/image/clip_search_top_7_wine.png)

![picture of wine using ResNet50](https://github.com/yuc0805/Image-to-Image-Search-Using-CLIP/blob/main/output_dir/resnet50/image/resnet50_search_top_7_wine.png)


## Credits

- [Kritik Seth](https://www.kaggle.com/kritikseth) for providing the dataset.
- Dependencies: `torch`, `PIL`, `numpy`, `tqdm`.

---
