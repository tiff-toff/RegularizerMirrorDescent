import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as datasets

import math
import numpy as np
import random

from models import *
from corrupted import *
import SMD_opt
import RMD_loss

from time import time
import os
import yaml

import argparse
from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section

Section('dataset', 'Loading data and applying corrupt').params(
    root=Param(str, 'cifar10 data directory', required=True),
    presaved=Param(bool, 'flag indicating whether to use presaved corrupted dataset', required=True),
    presaved_path=Param(str, 'path of presaved corrupted dataset'),
    prob=Param(float, 'corruption probability on training label', default=0.0),
    seed=Param(int, 'random seed for label noise', default=0)
)

Section('architecture', 'Architecture and initialization').params(
    init_path=Param(str, 'path to saved model with initialized parameters', default=None)
)

Section('training', 'Hyperparameters').params(
    qnorm=Param(float, 'q-norm', default=2),
    lr=Param(float, 'learning rate', required=True),
    lmbda=Param(float, 'lambda / weight decay parameter', default=0),
    epochs=Param(int, 'total number of epochs', required=True),
    history=Param(int, 'number of epochs to look at for stopping condition', default=25),
    batch_size=Param(int, 'batch size', default=128)
)

Section('output', 'Where to write outputs').params(
    output_directory=Param(str, 'directory to save ouptuts', required=True)
)

Section('checkpoint', 'Saving and training from checkpoints').params(
    from_checkpoint=Param(bool, 'whether to train from checkpoint', required=True),
    trial=Param(int, 'which trial number to grab checkpoint from', default=1)
)


@param('dataset.root')
@param('dataset.prob')
@param('dataset.seed')
@param('dataset.presaved')
@param('dataset.presaved_path')
@param('training.batch_size')
def make_dataset(root=None, prob=None, seed=None, presaved=None, presaved_path=None, batch_size=None):
    # Data
    print('==> Preparing data..')
    root = os.path.expandvars(root)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    if (presaved):
        class MyDataset(torch.utils.data.Dataset):
            def __init__(self):
                self.cifar10 = torch.load(presaved_path) #already transformed
                
            def __getitem__(self, index):
                data, target = self.cifar10[index]
                
                return data, target, index

            def __len__(self):
                return len(self.cifar10)

        trainset = MyDataset()
    else:
        trainset = CorruptedCIFAR10(prob=prob, seed=seed, return_index=True, root=root, train=True, download=False, transform=transform_train)   
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # check corruption levels (get indices)
    origset = torchvision.datasets.CIFAR10(root=root, train=True, download=False, transform=transform_test)

    uncorrupted_indices = []
    corrupted_indices = []
    for (img1, lab1, idx), (img2, lab2) in zip(trainset, origset):
        if lab1 != lab2:
            corrupted_indices.append(idx)
        else:
            uncorrupted_indices.append(idx)
    print('num corrupted:', len(corrupted_indices))
    print('num uncorrupted:', len(uncorrupted_indices))

    testset = torchvision.datasets.CIFAR10(root=root, train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader, len(trainset), len(testset), corrupted_indices, uncorrupted_indices


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.uniform_(-0.01, 0.01)
        m.bias.data.uniform_(-0.1, 0.1)

@param('architecture.init_path')
def create_model(device, init_path=None):
    # Model
    print('==> Building model..')
    print('using...', device)

    model = ResNet18()
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    model = model.cuda()
    
    if (init_path != None):
        init_path = os.path.expandvars(init_path)
        print('initialization:', init_path)
        model.load_state_dict(torch.load(init_path))
    else:
        model.apply(weights_init)
    
    free_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("free params:", free_params)
    return model


@param('training.epochs')
@param('training.qnorm')
@param('training.lr')
@param('training.lmbda')
@param('training.history')
@param('checkpoint.from_checkpoint')
@param('checkpoint.trial')
def train(device, model, trainloader, output_dir, N, corrupted_indices, uncorrupted_indices, epochs=None, qnorm=None, lr=None, lmbda=None, history=None, from_checkpoint=None, trial=None):
    
    z =  torch.normal(0.0, 0.0000005, size=(50000,), device=device).type(torch.float64) #small z
    per_sample_loss = torch.zeros(N, device=device)
    total_loss = torch.zeros(epochs, device=device)
    per_sample_criterion = nn.CrossEntropyLoss(reduction='none').to(device) #per sample losses
    optimizer = SMD_opt.SMD_qnorm(model.parameters(), lr=lr, q=qnorm)
    rmd_criterion = RMD_loss.RMD_Loss(per_sample_criterion, N) #losses are averaged in minibatch

    end_cond_count = 0
    new_constraint = 0
    old_constraint = 10**6
    training_info = {}
    start_epoch = 0
    if from_checkpoint:
        checkpoint = torch.load(f'{output_dir}/checkpoint.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        z = checkpoint['z_variables']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        end_cond_count = checkpoint['count']
        old_constraint = checkpoint['old_constraint']
        training_info = checkpoint['training_info']
        print(f'Training from epoch {start_epoch+1}...')
        
    print('\nTraining...')
    # Training
    for epoch in range(start_epoch, epochs):
        model.train()

        epoch_t = time()

        for images, labels, idx in trainloader:
            # training step
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            images = Variable(images)
            labels = Variable(labels)

            # Run the forward pass
            outputs = model(images)
            sample_loss = per_sample_criterion(outputs, labels)   # per-sample loss in each batch

            rmd_criterion.set_z_values(z, idx)
            rmd_loss = rmd_criterion(outputs, labels)

            # Backprop
            optimizer.zero_grad()
            rmd_loss.backward()
            optimizer.step()

            # Compute c_i for batch 
            c_batch = lr * (z[idx] - torch.sqrt(2*sample_loss.clone().detach()))
            c_batch = c_batch.clone().detach()

            # Update z-auxiliary variables
            z[idx] = z[idx] - (lmbda * c_batch)
        
        print('Epoch [{}/{}] Train Time: {:.2f}s'.format(epoch+1, epochs, time()-epoch_t))

        # evaluate loss and accuracy at end of epoch
        total = 0
        correct = 0
        model.eval()

        with torch.no_grad():
            for images, labels, idx in trainloader:
                idx = idx.squeeze()

                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                images = Variable(images)
                labels = Variable(labels)
                
                # evaluate and compute loss + accuracy
                outputs = model(images)
                per_sample_loss[idx] = per_sample_criterion(outputs, labels)      # per-sample loss

                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        total_loss[epoch] = per_sample_loss.sum().item()
        total_loss[epoch] /= N   # average loss of training set
        new_constraint = torch.sum(torch.abs(z - torch.sqrt(2*per_sample_loss.clone().detach())))

        if len(corrupted_indices) > 0 and len(uncorrupted_indices) > 0:
            corrupted_loss = per_sample_loss[corrupted_indices].sum().item() / len(corrupted_indices)
            uncorrupted_loss = per_sample_loss[uncorrupted_indices].sum().item() / len(uncorrupted_indices)
            print('Corrupted Loss: {}, Uncorrupted Loss: {}'.format(corrupted_loss, uncorrupted_loss))

        print('Epoch [{}/{}], Loss: {:.4f}, Constraint: {:.4f}, Accuracy: {:.2f}%, Time: {:.2f}s'
              .format(epoch + 1, epochs, total_loss[epoch], new_constraint, (correct / total) * 100, time()-epoch_t))

       # Exit condition: small change in constraint
        improvement = (old_constraint - new_constraint) / old_constraint
        print('Improvement: {}'.format(improvement))
        print('Total Loss: {}'.format(total_loss[epoch]))
        print('Constraint: {}'.format(new_constraint))
        w = torch.nn.utils.parameters_to_vector(model.parameters())
        q_norm_w = torch.sum(torch.abs(w)**qnorm)
        print('q-norm Weights: {}'.format(q_norm_w))
        z_squared = torch.sum(torch.abs(z)**2)
        print('z-squared: {}'.format(z_squared))

        if (improvement >= 0.0001):
            old_constraint = new_constraint
            end_cond_count = 0
            print("Count: 0")
            training_info = {'total_loss': total_loss[epoch].item(),
                             'q_norm': q_norm_w.item(),
                             'z_squared': z_squared.item(),
                             'train_acc': correct / total}
            if len(corrupted_indices) > 0 and len(uncorrupted_indices) > 0:
                    training_info['corrupted_loss'] = corrupted_loss
                    training_info['uncorrupted_loss'] = uncorrupted_loss

            torch.save(model.state_dict(), f'{output_dir}/final_model.pt')
            torch.save(z, f'{output_dir}/final_z.pt')
        else:
            end_cond_count += 1
            print("Count:", end_cond_count)
        
        if (end_cond_count == history):
            break
        
        # Checkpointing
        if (epoch + 1) % 25 == 0:
            print('Checkpointing...')
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'z_variables': z, 
                'optimizer_state_dict': optimizer.state_dict(),
                'count': end_cond_count,
                'old_constraint': old_constraint,
                'training_info': training_info
            }, f'{output_dir}/checkpoint.pt')
          
    print('Finished Training')
    torch.save(model.state_dict(), f'{output_dir}/last_trained_model.pt')
    torch.save(z, f'{output_dir}/last_trained_z.pt')
    return training_info


def evaluate(device, model, testloader):
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in testloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            images = Variable(images)
            labels = Variable(labels)
            
            outputs = model(images)
            predicted = torch.argmax(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))
    return correct / total


if __name__=="__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)

    config = get_current_config()
    parser = argparse.ArgumentParser(description='CIFAR-10 regularized training')
    config.augment_argparse(parser)

    # loads from args.config_path if provided
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()

    if config['checkpoint.from_checkpoint']:
        print("from checkpoint")
        os.environ["OUTPUT"] = "$GROUP/rmd-experiment/rmd/" + str(config['checkpoint.trial'])

    output_dir = os.path.expandvars(config['output.output_directory'])
    print(output_dir)
    output_dir = os.path.expandvars(output_dir)
    print(output_dir)
    

    if not config['checkpoint.from_checkpoint']:
        print("new training!")
        os.makedirs(output_dir, exist_ok=True)

    trainloader, testloader, N, _, corrupted_indices, uncorrupted_indices = make_dataset()
    model = create_model(device)
    # save initial model
    if not config['checkpoint.from_checkpoint']:
        torch.save(model.state_dict(), f'{output_dir}/init_model.pt')

    start_t = time()
    training_info = train(device, model, trainloader, output_dir, N, corrupted_indices, uncorrupted_indices)
    model.load_state_dict(torch.load(f'{output_dir}/final_model.pt'))
    test_acc = evaluate(device, model, testloader)
    print(f'Total time: {(time()-start_t)/3600:.2f} hrs')

    training_info['test_acc'] = test_acc
    result_file = f'{output_dir}/results.yaml'
    with open(result_file, 'w') as file:
        yaml.dump(training_info, file)
        
