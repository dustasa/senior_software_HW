import sys

import data
import argparse
from model import *


BATCH_SIZE = 128
MAXLEN = 100

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")


def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            print(
                '{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch, loss_train / len(train_loader)))


def validate(model, train_loader, val_loader):
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())
        print("Accuracy {}: {:.4f}".format(name, correct / total))


def train(model_type='Net',
          learning_rate=1e-2,
          epochs=2,
          batch_size=128):

    # train, val划分
    train_loader = torch.utils.data.DataLoader(data.cifar2, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(data.cifar2_val, batch_size=batch_size, shuffle=False)

    if model_type == 'Net':
        model = Net()
    elif model_type == 'NetWidth':
        model = NetWidth()
    elif model_type == 'NetDropout':
        model = NetDropout()
    elif model_type == 'NetBatchNorm':
        model = NetBatchNorm()
    elif model_type == 'NetRes':
        model = NetRes()
    elif model_type == 'NetResDeep':
        model = NetResDeep()
    elif model_type == 'BigTransfer':
        weights_cifar10 = get_weights('BiT-M-R50x1-CIFAR10')
        model = ResNetV2(ResNetV2.BLOCK_UNITS['r50'], width_factor=1, head_size=10)
        model.load_from(weights_cifar10)
    else:
        print("Invalid model_type : %s", model_type)
        return
    sys.stdout.flush()

    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), learning_rate, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    training_loop(
        epochs,
        optimizer,
        model,
        loss_fn,
        train_loader,
    )

    validate(model, train_loader, val_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="program description")
    parser.add_argument('-m', '--model', default='Net')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, choices=[0.01, 0.005, 0.003])
    parser.add_argument('-e', '--epoch', type=int, default=2, choices=[1, 2, 3, 4, 5, 10, 20, 50, 100])
    parser.add_argument('-bs', '--batch_size', type=int, default=128, choices=[32, 64, 128, 256, 512])

    args = parser.parse_args()

    print('model_name : %s' % args.model)
    print('learning_rate : %f' % args.learning_rate)
    print('epoch : %d' % args.epoch)
    print('batch_size : %d' % args.batch_size)

    train(args.model, args.learning_rate, args.epoch, args.batch_size)
