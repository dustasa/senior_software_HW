import torch.utils
from torchvision import transforms
from torchvision import datasets
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Downloading CIFAR-10
data_path = '../data-unversioned/p1ch7/'
cifar10 = datasets.CIFAR10(data_path, train=True, download=True)
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True)  # 下载太慢请开代理

# 引入normalize的数据初始化
tensor_cifar10_normalize_train = datasets.CIFAR10(data_path, train=True, download=False,
                                                  transform=transforms.Compose([
                                                      transforms.RandomHorizontalFlip(),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.4915, 0.4823, 0.4468),
                                                                           (0.2470, 0.2435, 0.2616))
                                                  ]))

tensor_cifar10_normalize_val = datasets.CIFAR10(data_path, train=False, download=False,
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.4915, 0.4823, 0.4468),
                                                                         (0.2470, 0.2435, 0.2616))
                                                ]))

# Build the dataset and DataLoader
label_map = {0: 0, 2: 1}  # 占位符
class_names = ['airplane', 'bird']

# 训练集
cifar2 = [(img, label_map[label])
          for img, label in tensor_cifar10_normalize_train
          if label in [0, 2]]

# 验证集
cifar2_val = [(img, label_map[label])
              for img, label in tensor_cifar10_normalize_val
              if label in [0, 2]]

