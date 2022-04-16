# Импорт необходимых библиотек
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.utils.data import ConcatDataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
from copy import deepcopy

# Сохранение значений точности и потерь во время обучения и тестирования для построение графика
result = {
    'val_acc': [],
    'train_acc': [],
    'val_loss': [],
    'train_loss': []
}


# Функция обучения модели
def train_model(model, entropy_loss, optimizer, epochs):
    start_time = time.time()
    best_model = deepcopy(model.state_dict())
    best_accuracy = 0.0

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')

        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                preds = torch.max(outputs, 1)[1]
                loss = entropy_loss(outputs, labels)
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        train_loss = running_loss / train_size
        train_acc = running_corrects.double() / train_size
        print(f'Train Loss: {train_loss} Train Acc: {train_acc}')
        result['train_acc'].append(train_acc)
        result['train_loss'].append(train_loss)

        model.eval()  # Отключение вычисления градиентов для оценки модели
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = entropy_loss(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        val_loss = running_loss / val_size
        val_acc = running_corrects.double() / val_size
        print(f'Val Loss: {val_loss} Val Acc: {val_acc}')
        result['val_loss'].append(val_loss)
        result['val_acc'].append(val_acc)

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_model = deepcopy(model.state_dict())

    end_time = time.time() - start_time
    print(f'Training complete in {end_time // 60}m {int(end_time % 60)}s')
    print(f'Best Val Acc: {best_accuracy}')
    model.load_state_dict(best_model)
    return model


# Тестирование модели
def test_model(model, dataset):
    labels = []
    predictions = []
    accuracy = 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            img, label = dataset[i]
            img = img[np.newaxis, :]
            img = img.to(device)
            output = model(img)
            preds = torch.max(output, 1)[1]
            if class_names[label] == class_names[preds]:
                accuracy += 1
            labels.append(class_names[label])
            predictions.append(class_names[preds])
    print(f'Test Accuracy: {accuracy / len(dataset)}')
    model.train(mode=was_training)  # Возврат в режим обучения
    return labels, predictions


if __name__ == '__main__':
    train = transforms.Compose([
        transforms.Resize(500),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train2 = transforms.Compose([
        transforms.Resize(500),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test = transforms.Compose([
        transforms.Resize(500),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_set = torchvision.datasets.ImageFolder('chest_xray/train',
                                                 transform=train)
    train_set2 = torchvision.datasets.ImageFolder('chest_xray/train',
                                                  transform=train2)
    test_set = torchvision.datasets.ImageFolder('chest_xray/test', transform=test)
    val_set = torchvision.datasets.ImageFolder('chest_xray/val', transform=test)

    train_size = len(train_set)
    test_size = len(test_set)
    val_size = len(val_set)
    print(f'Train: {train_size} images, Test: {test_size} images, Validation: {val_size} images')

    class_names = train_set.classes

    train_set2 = train_test_split(train_set2, test_size=3875 / train_size, shuffle=False)
    train_set = ConcatDataset([train_set, train_set2])

    train_set, val_set = train_test_split(train_set, test_size=0.2)

    train_size = len(train_set)
    val_size = len(val_set)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=True, num_workers=2)

    model = torchvision.models.resnet50(pretrained=True)

    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(class_names))

    # Параметры
    entropy_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 50

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model = train_model(model, entropy_loss, optimizer, num_epochs)  # Обучение модели

    labels, predictions = test_model(model, test_set)

    torch.save(model, 'model.pb')

    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax = ax.ravel()

    for i, res in enumerate(['acc', 'loss']):
        ax[i].plot(result['train_' + res])
        ax[i].plot(result['val_' + res])
        ax[i].set_title(f'Model {res}')
        ax[i].set_xlabel('epochs')
        ax[i].set_ylabel(res)
        ax[i].legend(['train', 'val'])
    plt.show()
