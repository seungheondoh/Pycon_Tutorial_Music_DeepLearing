'''
train_test.py

A file for training model for genre classification.
Please check the device in hparams.py before you run this code.
'''
import torch
import data_manager
import models
from hparams import hparams

# Wrapper class to run PyTorch model
class Runner(object):
    def __init__(self, hparams):
        # __init__에 model 을 올려줍시다
        self.model = models.Baseline(hparams)
        # Loss function은 CrossEntropyLoss()를 호출해줍시다.
        self.criterion = torch.nn.CrossEntropyLoss()
        # Stocastic Gradient Descent입니다
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=hparams.learning_rate)
        # Learning_rate도 저장해줍시다
        self.learning_rate = hparams.learning_rate
        # 기본 디바이스는 이번에는 cpu!
        self.device = torch.device("cpu")

        # 혹시라도 GPU를 가지고 계신분들을 위해서
        if hparams.device > 0:
            torch.cuda.set_device(hparams.device - 1)
            # model.cuda
            self.model.cuda(hparams.device - 1)
            # criterion.cuda
            self.criterion.cuda(hparams.device - 1)
            # device.cuda를 지정해줍시다
            self.device = torch.device("cuda:" + str(hparams.device - 1))

    # Accuracy function works like loss function in PyTorch
    def accuracy(self, source, target):
        # 예측치와 실제 값을 비교하기 위해서 해줍시다.
        source = source.max(1)[1].long().cpu()
        # GPU를 사용하더라도 accuracy연산은 cpu로 해도 괜찮습니다.
        target = target.cpu()
        # .tiem()은 tensor를 python number은 꺼내줍니다.
        correct = (source == target).sum().item()

        return correct/float(source.size(0))

    # Running model for train, test and validation. mode: 'train' for training, 'eval' for validation and test
    def run(self, dataloader, mode='train'):
        self.model.train() if mode is 'train' else self.model.eval()

        epoch_loss = 0
        epoch_acc = 0
        for batch, (x, y) in enumerate(dataloader):
            x = x.to(self.device)
            y = y.to(self.device)

            prediction = self.model(x)
            loss = self.criterion(prediction, y)
            acc = self.accuracy(prediction, y)

            if mode is 'train':
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            epoch_loss += prediction.size(0)*loss.item()
            epoch_acc += prediction.size(0)*acc

        epoch_loss = epoch_loss/len(dataloader.dataset)
        epoch_acc = epoch_acc/len(dataloader.dataset)

        return epoch_loss, epoch_acc


def device_name(device):
    if device == 0:
        device_name = 'CPU'
    else:
        device_name = 'GPU:' + str(device - 1)

    return device_name

def main():
    train_loader, valid_loader, test_loader = data_manager.get_dataloader(hparams)
    runner = Runner(hparams)

    print('Training on ' + device_name(hparams.device))
    for epoch in range(hparams.num_epochs):
        train_loss, train_acc = runner.run(train_loader, 'train')
        valid_loss, valid_acc = runner.run(valid_loader, 'eval')

        print("[Epoch %d/%d] [Train Loss: %.4f] [Train Acc: %.4f] [Valid Loss: %.4f] [Valid Acc: %.4f]" %
              (epoch + 1, hparams.num_epochs, train_loss, train_acc, valid_loss, valid_acc))

    test_loss, test_acc = runner.run(test_loader, 'eval')
    print("Training Finished")
    print("Test Accuracy: %.2f%%" % (100*test_acc))

if __name__ == '__main__':
    main()
