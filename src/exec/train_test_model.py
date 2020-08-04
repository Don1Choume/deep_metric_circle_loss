from pathlib import Path
import pickle
import cloudpickle
from tqdm import tqdm
import sys
current_dir = Path(__file__).resolve().parent
sys.path.append(str(Path(str(current_dir) + '/../')))

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.optim import SGD
from torch.utils.data import DataLoader
from models.CNN_model import Encoder, CosLayer, Classifier
from models.loss_func import CircleLoss

from torchsummary import summary

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
    epoch_num = 2

    train_loader = DataLoader(get_dataset('train'), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(get_dataset('valid'), batch_size=1000, shuffle=False)

    model = Encoder().to(device)
    classifier = CosLayer(64, 10, loss_type='softmax').to(device)
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
    optimizer_cls = SGD(classifier.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
    criterion = CircleLoss(m=0.25, gamma=80, similarity='cos').to(device)
    criterion_cls = nn.CrossEntropyLoss().to(device)

    print(model)
    summary(model, (1, 28, 28))

    model.train()
    for epoch in range(epoch_num):
        print('Train Encoder, Epoch {0}/{1}'.format(epoch + 1, epoch_num))
        for img, label in tqdm(train_loader, total=len(train_loader)):
            img = img.to(device)
            label = label.to(device)
            model.zero_grad()
            features = model(img)
            loss = criterion(features, label)
            loss.backward()
            optimizer.step()

    classifier.train()
    for epoch in range(epoch_num):
        print('Train Classifier, Epoch {0}/{1}'.format(epoch + 1, epoch_num))
        for img, label in tqdm(train_loader, total=len(train_loader)):
            img = img.to(device)
            label = label.to(device)
            model.zero_grad()
            classifier.zero_grad()
            features = model(img)
            output = classifier(features.reshape(-1, 64))
            loss = criterion_cls(output, label)
            loss.backward()
            optimizer_cls.step()

    save_model(model, 'circle_loss_Encoder.pkl')
    save_model(classifier, 'circle_loss_Classifier.pkl')

    model.eval()
    classifier.eval()
    correct = 0
    print('Test Classifier')
    for img, label in tqdm(val_loader, total=len(val_loader)):
        with torch.no_grad():
            img = img.to(device)
            label = label.to(device)
            pred = classifier(model(img).reshape(-1, 64)).data.max(1)[1]
            correct += pred.eq(label.data).cpu().sum()

    print('Accuracy: {}/{} ({:.0f}%)'.format(
        correct, len(val_loader.dataset), 100. * correct / len(val_loader.dataset)))


if __name__ == "__main__":
    train_classifier()