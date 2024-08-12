import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

### Loading the pretrained model
resnet = models.resnet18(pretrained = True)


## print the model architecture
# print(resnet)


### Modifye the model to extract features

## To use the ResNet model as a feature encoder, we can remove 
## final fully connected layer that is used for classification and
## keep the earlier layers.

class featureExtractor(nn.Module):
    def __init__(self, original_model):
        super(featureExtractor, self).__init__()
        ## keep all layers except the final fully connected layer
        self.features = nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) ### Flatten the output for further processing
        return x

### Instantiate the feature extractor
feature_extractor  = featureExtractor(resnet)

print(" feature extractor is as follows:")
# print(feature_extractor)




### Prepare an input image

## Lets load and preprocess an image to pass through the feature encoder

image = Image.open('/home/dheeraj/Downloads/aesthetic-macbook-purple-sky-ztaeqfld1emy3v09.jpg')
# image = np.array(image, dtype = float)

print(" image is : ", image)
transform = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                     std = [0.229, 0.224, 0.225]),
                                ])


### apply the transformations to the image
input_tensor = transform(image).unsqueeze(0) ## Add batch dimension

# print(" length of input tensor = ", input_tensor)
print(" shape of input tensor: ".format(input_tensor.shape))



### Extract features

## Pass the preprocessed image through the feature extractor to get the encoded features

with torch.no_grad():
    features = feature_extractor(input_tensor)

## output the shape of the extracted features
print("features are : ", features.shape) ### Should output torc.Size([1, 512])
