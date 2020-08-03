from pathlib import Path
import pickle
import cloudpickle
import sys
current_dir = Path(__file__).resolve().parent
sys.path.append(str(Path(str(current_dir) + '/../')))

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.optim import SGD
from torch.utils.data import DataLoader
from models.CNN_model import Encoder, Classifier
from models.loss_func import CosLayer, CircleLoss

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

def get_dataset(data_type='train'):
    project_dir = Path(__file__).resolve().parents[2]
    data_path = project_dir/'data'/'processed'
    if data_type=='train':
        data_name = 'train_dataset.pkl'
    else:
        data_name = 'valid_dataset.pkl'
    with open(str(data_path/data_name), 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def save_model(model, savename):
    project_dir = Path(__file__).resolve().parents[2]
    model_path = project_dir/'models'
    with open(str(model_path/savename), 'wb') as f:
        cloudpickle.dump(model, f)

def load_model(savename):
    project_dir = Path(__file__).resolve().parents[2]
    model_path = project_dir/'models'
    with open(str(model_path/savename), 'rb') as f:
        model = cloudpickle.load(f)
    return model

def train_classifier():
    batch_size = 64
    epoch_num = 20

    model = Encoder()
    classifier = Classifier()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
    optimizer_cls = SGD(classifier.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
    train_loader = DataLoader(get_dataset('train'), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(get_dataset('valid'), batch_size=1000, shuffle=False)
    criterion = CircleLoss(m=0.25, gamma=80, similarity='cos')
    criterion_xe = nn.CrossEntropyLoss()

    for epoch in range(epoch_num):
        for img, label in train_loader:
            model.zero_grad()
            features = model(img)
            loss = criterion(features, label)
            loss.backward()
            optimizer.step()
        print('Train Encoder, Epoch {0}/{1}'.format(epoch + 1, epoch_num))

    for epoch in range(20):
        for img, label in train_loader:
            model.zero_grad()
            classifier.zero_grad()
            features = model(img)
            output = classifier(features)
            loss = criterion_xe(output, label)
            loss.backward()
            optimizer_cls.step()
        print('Train Classifier, Epoch {0}/{1}'.format(epoch + 1, epoch_num))

    save_model(model, 'circle_loss_Encoder.pkl')
    save_model(classifier, 'circle_loss_Classifier.pkl')

    tp = 0
    fn = 0
    fp = 0
    thresh = 0.75
    for img, label in val_loader:
        pred = model(img)
        gt_label = label[0] == label[1]
        pred_label = torch.sum(pred[0] * pred[1]) > thresh
        if gt_label and pred_label:
            tp += 1
        elif gt_label and not pred_label:
            fn += 1
        elif not gt_label and pred_label:
            fp += 1

    print("Recall: {:.4f}".format(tp / (tp + fn)))
    print("Precision: {:.4f}".format(tp / (tp + fp)))


if __name__ == "__main__":
    train_classifier()