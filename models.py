from torch import nn
import torchvision



# generic fully-connected neural network class
class NeuralNetwork(nn.Module):

    # layers = int layer sizes, starting with the input layer
    def __init__(self, *layers):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()

        stack = []
        for i in range(len(layers) - 1):
            stack.extend([nn.Linear(layers[i], layers[i + 1]), nn.ReLU()])

        self.linear_relu_stack = nn.Sequential(*stack[:-1])

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits



class VGGNet(nn.Module):

    def __init__(self, device ='cpu'):
        super(VGGNet, self).__init__()

        self.model = torchvision.models.vgg19(pretrained=True).to(device)

        #freeze all but last layer
        last_layer_idx = 34
        for layer in list(self.model.features.children())[:last_layer_idx]:
            for param in layer.parameters():
                param.requires_grad = False

        self.model.classifier = nn.Sequential(
          nn.Linear(in_features=25088, out_features=512, bias=True, device=device),
          nn.ReLU(inplace=True),
          nn.Dropout(p=0.5, inplace=False),
          nn.Linear(in_features=512, out_features=36, bias=True, device=device),
          nn.ReLU(inplace=True),
          nn.Dropout(p=0.5, inplace=False),
          nn.Linear(in_features=36, out_features=1, bias=True, device=device)
        )

    def forward(self, x):
        return self.model(x).squeeze()

class ResNet18(nn.Module):

    def __init__(self, device='cpu', pretrained=True, freeze=True):
        super(ResNet18, self).__init__()

        self.model = torchvision.models.resnet18(pretrained=pretrained).to(device)
        
        if pretrained and freeze:
            for param in self.model.parameters():
                param.requires_grad = False


            for param in self.model.layer4.parameters():
                param.requires_grad = True

        self.model.fc = nn.Sequential(
          nn.Linear(in_features=512, out_features=36, bias=True, device=device),
          nn.ReLU(inplace=True),
          nn.Dropout(p=0.5, inplace=False),
          nn.Linear(in_features=36, out_features=1, bias=True, device=device)
        )

    def forward(self, x):
        return self.model(x).squeeze()


   