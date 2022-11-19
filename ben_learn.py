
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.image as img
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

# device settings -- if GPU doesn't work on WSL type in terminal: export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('using gpu')
# device = torch.device('cpu')
# path to training data
train_path = r'/train_test_data/train'
# classification labels
label = ['0','1','2']

# read in metadata as dataframe -- this contains image paths and classfications
df = pd.read_csv(r'new_train.csv')
# print(data)
# mask = df['label']==0
# all_0 = df[mask]
# mask = df['label']==1
# all_1 = df[mask]
# mask = df['label']==2
# all_2 = df[mask]


# hyperparameters
num_epochs = 35
num_classes = 2
batch_size = 25
learning_rate = 0.001

# split dataframe randomly into training data and validation data
train, valid_data = train_test_split(df, stratify=df['label'], test_size=0.2)

# define dataset class particular to my data
class ForestDataset(Dataset):
    def __init__(self, df, path , transform = transforms.functional.to_tensor):
        super().__init__()
        self.df = df
        self.path = path
        self.transform = transform
        self.labels = df['label']
    def __len__(self):
        return len(self.df)

    def __getitem__(self,index):
        row = self.df.iloc[index]
        img_name = row['example_path'].split('/')[-1]
        img_path = os.path.join(os.getcwd()+self.path, img_name)
        image = img.imread(img_path)
        label = row['label']
        if self.transform is not None:
            image = self.transform(image)
            image_sm = transforms.Resize(32)(image)
        return image_sm, label

# make datasets
train_dataset = ForestDataset(train, train_path)
valid_dataset = ForestDataset(valid_data, train_path)
test_dataset = ForestDataset
# load data 
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size = batch_size, shuffle=False, num_workers=0)


# display images (doesn't work!)
def imshow(t_image, ax=None, title=None, normalize=False):
    if ax is None:
        fig, ax = plt.subplots()
    # np_image = image.np().transpose((1, 2, 0))
    image = transforms.ToPILImage()(t_image).convert("RGB")
    image_sm = transforms.Resize(32)(image)
    if normalize:
        np_image = t_image.numpy()
        mean = np_image.mean()
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)


    ax.imshow(image_sm)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    return ax

trainimages, trainlabels = next(iter(train_loader))
def plot_images(trainimages, trainlabels):
    fig, axes = plt.subplots(figsize=(12, 12), ncols=5)
    print('training images')
    for i in range(5):
        axe1 = axes[i] 
        imshow(trainimages[i], ax=axe1, normalize=False)
    print(trainimages[0].size())
    print(trainlabels)


plot_images(trainimages,trainlabels)

# make neural network
class CNN(nn.Module): 
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(720, 1024)
        self.fc2 = nn.Linear(1024, 3)


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


# optimization settings
model = CNN().to(device)  # -- from tutorial https://www.pluralsight.com/guides/image-classification-with-pytorch

# from pytorch website -- https://pytorch.org/tutorials/advanced/neural_style_tutorial.html?highlight=image%20classify

# cnn = models.vgg19(pretrained=True).features.to(device).eval()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


# keeping-track-of-losses 
train_losses = []
valid_losses = []

def traintime():
    # let's go!
    print('Start training for %s epochs!' % num_epochs)
    for epoch in range(1, num_epochs + 1):
        # keep-track-of-training-and-validation-loss
        train_loss = 0.0
        valid_loss = 0.0

        # training-the-model
        for data, target in train_loader:
            # move-tensors-to-GPU 
            data = data.to(device)
            target = target.to(device)
            # clear-the-gradients-of-all-optimized-variables
            optimizer.zero_grad()
            # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
            output = model(data)
            # calculate-the-batch-loss
            loss = criterion(output, target)
            # backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
            loss.backward()
            # perform-a-ingle-optimization-step (parameter-update)
            optimizer.step()
            # update-training-loss
            train_loss += loss.item() * data.size(0)
        # validate-the-model
        model.eval()

        for data, target in valid_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output, target)
            # update-average-validation-loss 
            valid_loss += loss.item() * data.size(0)
        # calculate-average-losses
        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(valid_loader.sampler)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        # print-training/validation-statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))




traintime()


def test_model():
    # test-the-model
    model.eval()  # it-disables-dropout
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Test Accuracy of the model: {} %'.format(100 * correct / total))


# Save 

torch.save(model.state_dict(), 'model.ckpt')
    # # ==================================================================================

    # # DAN's TUTORIAL


