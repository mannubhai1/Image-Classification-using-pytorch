import torch
from PIL import Image
from torch import nn, load, save
from torch.optim import Adam    
from torch.utils.data import DataLoader     
from torchvision import datasets       
from torchvision.transforms import ToTensor

train = datasets.MNIST(root = 'data', download = True,train=True,transform = ToTensor())        # Load the dataset (train
dataset = DataLoader(train,32)      # Create a data loader


class ImageClassifier(nn.Module):       # Define the model
    def __init__(self):      
        super().__init__()   
        self.model = nn.Sequential(    # Define the layers
            nn.Conv2d(1,32,(3,3)),
            nn.ReLU(),
            nn.Conv2d(32,64,(3,3)),
            nn.ReLU(),
            nn.Conv2d(64,64,(3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6),10)
        )

    def forward(self,x):        # Forward pass
        return self.model(x)

# Instances of the model
clf = ImageClassifier().to('cpu')
opt = Adam(clf.parameters(),lr = 0.001)
loss_fn = nn.CrossEntropyLoss()


if __name__ == '__main__':  

    with open('model.pt','rb') as f:  
        clf.load_state_dict(load(f))  # Load the model
        img = Image.open('img_3.jpg')          #image name
        img_tensor = ToTensor()(img).unsqueeze(0).to('cpu')   # Convert the image to a tensor
        print(torch.argmax(clf(img_tensor)))   # Print the prediction

    # for epoch in range(10):
    #     for batch in (dataset):
    #         X,y = batch # Get batch
    #         X,y = X.to('cpu'), y.to('cpu')  # Move the data to the device that is used
    #         yhat = clf(X)    # Compute the predictions
    #         loss = loss_fn(yhat,y)  # Compute the loss

    #         opt.zero_grad()     # Reset the gradients
    #         loss.backward()     # Compute the gradients
    #         opt.step()          # Update the parameters
        
    #     print(f"epoch : {epoch}, loss is : {loss.item()}")

    # with open('model.pt','wb') as f:
    #     save(clf.state_dict(),f)