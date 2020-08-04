from pathlib import Path
import pickle
import cloudpickle
from tqdm import tqdm
import numpy as np
import pandas as pd
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
device = torch.device('cuda' if cuda_available else 'cpu')
print('executed with '+('cuda' if cuda_available else 'cpu'))
loop_iter = 20

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
    torch.save(model.state_dict(), str(model_path/savename))
    # with open(str(model_path/savename), 'wb') as f:
    #     cloudpickle.dump(model, f)

def load_model(model, savename):
    project_dir = Path(__file__).resolve().parents[2]
    model_path = project_dir/'models'
    model.load_state_dict(torch.load(str(model_path/savename)))
    # with open(str(model_path/savename), 'rb') as f:
    #     model = cloudpickle.load(f)
    return model

def no_train():
    batch_size = 1000
    val_loader = DataLoader(get_dataset('valid'), batch_size=batch_size, shuffle=False)
    feats = []
    labels = []
    for img, label in tqdm(val_loader, total=len(val_loader)):
        with torch.no_grad():
            img = img.to(device).reshape(batch_size, -1)
            label = label.to(device).reshape(batch_size, -1)
            feats.append(img.to('cpu').detach().numpy().copy())
            labels.append(label.to('cpu').detach().numpy().copy())

    project_dir = Path(__file__).resolve().parents[2]
    rslt_path = project_dir/'result'
    print(np.concatenate(feats).shape)
    print(np.concatenate(labels).shape)
    np.save(str(rslt_path/'no_train_feat.npy'), np.concatenate(feats))
    np.save(str(rslt_path/'no_train_label.npy'), np.concatenate(labels))


def train_circleloss():
    batch_size = 64
    epoch_num = loop_iter

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

    save_model(model, 'circle_loss_Encoder.pth')
    save_model(classifier, 'circle_loss_Classifier.pth')

    model.eval()
    classifier.eval()
    correct = 0
    feats = []
    print('Test Classifier')
    for img, label in tqdm(val_loader, total=len(val_loader)):
        with torch.no_grad():
            img = img.to(device)
            label = label.to(device)
            feat = model(img).reshape(-1, 64)
            feats.append(feat.to('cpu').detach().numpy().copy())
            pred = classifier(feat).data.max(1)[1]
            correct += pred.eq(label.data).cpu().sum()

    print('Accuracy: {}/{} ({:.0f}%)'.format(
        correct, len(val_loader.dataset), 100. * correct / len(val_loader.dataset)))
    project_dir = Path(__file__).resolve().parents[2]
    rslt_path = project_dir/'result'
    np.save(str(rslt_path/'circleloss_feat.npy'), np.concatenate(feats))
    return correct, len(val_loader.dataset)


def train_cosloss(loss_type):
    batch_size = 64
    epoch_num = loop_iter

    train_loader = DataLoader(get_dataset('train'), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(get_dataset('valid'), batch_size=1000, shuffle=False)

    model = Encoder().to(device)
    classifier = CosLayer(64, 10, loss_type=loss_type).to(device)

    optimizer = SGD(list(model.parameters())+list(classifier.parameters()),
                    lr=0.001, momentum=0.9, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss().to(device)

    print(model)
    summary(model, (1, 28, 28))

    model.train()
    classifier.train()
    for epoch in range(epoch_num):
        print('Train Classifier, Epoch {0}/{1}'.format(epoch + 1, epoch_num))
        for img, label in tqdm(train_loader, total=len(train_loader)):
            img = img.to(device)
            label = label.to(device)
            model.zero_grad()
            classifier.zero_grad()
            features = model(img)
            if loss_type=='softmax':
                output = classifier(features.reshape(-1, 64))
            else:
                output = classifier(features.reshape(-1, 64), label)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

    save_model(model, loss_type+'_Encoder.pth')
    save_model(classifier, loss_type+'_Classifier.pth')

    model.eval()
    classifier.eval()
    correct = 0
    feats = []
    print('Test Classifier')
    for img, label in tqdm(val_loader, total=len(val_loader)):
        with torch.no_grad():
            img = img.to(device)
            label = label.to(device)
            feat = model(img).reshape(-1, 64)
            feats.append(feat.to('cpu').detach().numpy().copy())
            pred = classifier(feat).data.max(1)[1]
            correct += pred.eq(label.data).cpu().sum()

    print('Accuracy: {}/{} ({:.0f}%)'.format(
        correct, len(val_loader.dataset), 100. * correct / len(val_loader.dataset)))
    project_dir = Path(__file__).resolve().parents[2]
    rslt_path = project_dir/'result'
    np.save(str(rslt_path/(loss_type+'_feat.npy')), np.concatenate(feats))
    return correct, len(val_loader.dataset)


def train_funcs(idx):
    if idx==0:
        c,a = train_circleloss()
    elif idx==1:
        c,a = train_cosloss('softmax')
    elif idx==2:
        c,a = train_cosloss('sphereface')
    elif idx==3:
        c,a = train_cosloss('arcface')
    elif idx==4:
        c,a = train_cosloss('cosface')
    elif idx==5:
        c,a = train_cosloss('adacos')
    elif idx==6:
        c,a = train_cosloss('all')
    return c, a

if __name__ == "__main__":
    no_train()
    project_dir = Path(__file__).resolve().parents[2]
    rslt_path = project_dir/'result'

    iteration = 20
    corrects = [[] for i in range(7)]
    alls = [[] for i in range(7)]
    for i in range(iteration):
        for idx in range(7):
            c,a = train_funcs(idx)
            corrects[idx].append(c.to('cpu').detach().numpy().copy())
            alls[idx].append(a)
    pd.DataFrame(corrects, index=[
        'circleloss',
        'softmax',
        'sphereface',
        'arcface',
        'cosface',
        'adacos',
        'all']).to_csv(str(rslt_path/'all_correct.csv'))
    pd.DataFrame(alls, index=[
        'circleloss',
        'softmax',
        'sphereface',
        'arcface',
        'cosface',
        'adacos',
        'all']).to_csv(str(rslt_path/'all_all.csv'))