import os
import clip
import torch
import numpy as np
import pathlib
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import cv2 as cv
from evaluation.utils import setup_logging

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/16', device)


class Hotdog(Dataset):
    def __init__(self, path):
        data_root = pathlib.Path(path)
        all_image_paths = list(data_root.glob('*/*'))
        self.all_image_paths = [str(path) for path in all_image_paths]
        label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
        label_to_index = dict((label, index) for index, label in enumerate(label_names))
        self.all_image_labels = [label_to_index[path.parent.name] for path in all_image_paths]
        self.mean = np.array(mean).reshape((1, 1, 3))
        self.std = np.array(std).reshape((1, 1, 3))

    def __getitem__(self, index):
        img = cv.imread(self.all_image_paths[index])
        img = cv.resize(img, (224, 224))
        img = img / 255.
        img = (img - self.mean) / self.std
        img = np.transpose(img, [2, 0, 1])
        label = self.all_image_labels[index]
        img = torch.tensor(img, dtype=torch.float32)

        label = torch.tensor(label)
        return img, label

    def __len__(self):
        return len(self.all_image_paths)


def setup(args):
    # setup output directory and logging
    if args.output_dir[-6:] in 'exp_output':
        current_time = datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
        args.output_dir = f"{args.output_dir}/{args.model_name}/{args.dataset}/{current_time}"
        os.makedirs(args.output_dir)
    logger = setup_logging(args.output_dir)
    logger.info(args)
    return logger


def get_features(dataset):
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=1)):
            features = model.encode_image(images.to(device))
            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()


def main():
    train_data_path = 'your train data path'
    test_data_path = 'your test data path'
    train_data = Hotdog(train_data_path)
    test_data = Hotdog(test_data_path)
    train_features, train_labels = get_features(train_data)
    test_features, test_labels = get_features(test_data)
    classifier = LogisticRegression(random_state=0, C=0.316, max_iter=10000, verbose=1)
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)
    accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
    print(f"Accuracy = {accuracy:.3f}")


if __name__ == '__main__':
    main()
