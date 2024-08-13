import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os


### Define a custom dataset
## Lets create a custom dataset that includes images, classification labelsm and regression targets

class CustomDataset(Dataset):
    def __init__(self, image_dir, annotations, transform = None):
        self.image_dir = image_dir
        self.annotations  = annotations
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_name, class_label, reg_target = self.annotations[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, class_label, reg_target

## Example annotation list:
annotations = [
    ('image1.jpg', 0.0, 1.5),
    ('image2.jpg', 1.0, 3.2),
    ('image6.jpg', 0.0, 2.1)
    ### you can add more example if you want
]


### Define a multi task modek
### We'll build a model that includes the ResNet as feature encoder and 
### 2 separate heads for classification and regression


class MultiTaskModel(nn.Module):
    def __init__(self, base_model):
        super(MultiTaskModel, self).__init__()

        ## Use the pretrained model ResNet as the feature extractor
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])

        ### Classification head
        self.classifier = nn.Linear(512, 10)  ## Assuming 10 classes
        

        ## Regression head
        self.regressor = nn.Linear(512, 1) ### Single regressor output

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  ## Flatten the features obtained

        ## Pass the features to both the heada
        class_outputs = self.classifier(x)
        reg_outputs = self.regressor(x)

        return class_outputs, reg_outputs
    



### Prepare the dataset and DataLoader

transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),])



### Create the dataset and dataloader

dataset = CustomDataset(image_dir = "images_random", annotations = annotations, transform = transform)

dataloader = DataLoader(dataset, batch_size = 8, shuffle = True)
print("dataloader loaded", len(dataloader))


## Initialize and train the model


## Load the pretrained model
resnet = models.resnet18(pretrained = True)

model = MultiTaskModel(resnet)

print("model is : ", model)

## Define the loss functions and optimizer
criterion_class = nn.CrossEntropyLoss()
criterion_reg = nn. MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

## Training loop
model.train()

print(" dataloader is : ", dataloader)

for epoch in range(100):
    for images, class_labels, reg_targets in dataloader:
        
        ## Zero the parameters gradients
        optimizer.zero_grad()

        ## forward pass
        class_preds, reg_preds = model(images)
        print(" class prediction floating type :", class_preds.dtype)
        print("class_ labels dtype : ", class_labels.dtype)
        print(" ref  prediction floating type :", reg_preds.dtype)
        print("reg labels dtype : ", reg_targets.dtype)
        
        ## Compute the losses
        loss_class = criterion_class(class_preds,  class_labels.long())
        loss_reg = criterion_reg(reg_preds.squeeze(), reg_targets.float())


        ## Combine the losses ( you can weight them if needed)
        loss = loss_reg
        # loss = loss_class + loss_reg

        ## Backward pass and optimize
        loss.backward()
        optimizer.step()

    print("Epoch : {}, Loss : {}".format(epoch+1, loss.item()))




