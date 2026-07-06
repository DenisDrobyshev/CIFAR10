# Классификация изображений CIFAR-10

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)

Учебный проект по компьютерному зрению: классификация изображений датасета **CIFAR-10** (60 000 цветных картинок 32×32 в 10 классах). В репозитории сравниваются два подхода — свёрточная сеть как экстрактор признаков + классические алгоритмы, и обучение ResNet-18 «с нуля».

## Ноутбуки

### 1. `CIFAR10 PyTorch CNN LDA LogisticRegression.ipynb`
Гибридный пайплайн:
- свёрточная сеть на **PyTorch** обучается и используется как **экстрактор признаков**;
- извлечённые эмбеддинги (`train_features.csv`, `test_features.csv`) подаются в классические модели — **LDA** и **логистическую регрессию** (scikit-learn);
- сравнение качества CNN-классификатора и классических моделей поверх признаков.

### 2. `CIFAR10 ResNet18 Scratch.ipynb`
Реализация архитектуры **ResNet-18 с нуля** на PyTorch (остаточные блоки, downsample, BatchNorm), обучение на CIFAR-10. Обученные веса сохранены в `checkpoint/resnet18_cifar10.pth`.

## Структура

```
CIFAR10/
├── CIFAR10 PyTorch CNN LDA LogisticRegression.ipynb
├── CIFAR10 ResNet18 Scratch.ipynb
├── checkpoint/resnet18_cifar10.pth   # веса обученной ResNet-18
├── cifar_net.pth                     # веса CNN
├── train_features.csv / test_features.csv  # признаки, извлечённые CNN
└── data/cifar-10-batches-py/         # датасет CIFAR-10
```

## Запуск

```bash
git clone https://github.com/DenisDrobyshev/CIFAR10.git
cd CIFAR10
pip install torch torchvision scikit-learn pandas numpy matplotlib jupyter
jupyter notebook
```

Датасет уже включён в `data/`, поэтому ноутбуки запускаются без дополнительной загрузки.

## Стек

`PyTorch` · `torchvision` · `scikit-learn` (LDA, LogisticRegression) · `pandas` · `NumPy` · `matplotlib`
