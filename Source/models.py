import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class LeNet5(nn.Module):
    def __init__(self, num_classes,channels=1):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(channels, 6*channels, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6*channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6*channels, 16*channels, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16*channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(400*channels, 120*channels)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120*channels, 84*channels)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84*channels, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Residual block
class BasicBlock(nn.Module):
    def __init__(self,in_features=64,out_features=64,stride=[1,1],down_sample=False):
        # stride : list 
        # the value at corresponding indices are the strides of corresponding layers in a residual block
        
        super(BasicBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_features,out_features,3,stride[0],padding=1,bias=False) #weight layer
        self.bn1 = nn.BatchNorm2d(out_features) #weight layer
        
        self.relu = nn.ReLU(True) #relu
        
        self.conv2 = nn.Conv2d(out_features,out_features,3,stride[1],padding=1,bias=False) #weight layer
        self.bn2 = nn.BatchNorm2d(out_features) #weight layer

        self.down_sample = down_sample
        if down_sample:
            self.downsample = nn.Sequential(
                    nn.Conv2d(in_features,out_features,1,2,bias=False),
                    nn.BatchNorm2d(out_features)
                )

    def forward(self,x):
        x0=x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.down_sample:
            x0 = self.downsample(x0)  
        x = x + x0    # F(x)+x
        x= self.relu(x)
        return x

class Bottleneck(nn.Module):
    def __init__(self,inFeatures=64,outFeatures=64,kSize=[1,3,1],stride=[1,2,1],
    dn_sample=False,dnSample_stride=1) -> None:
        super(Bottleneck,self).__init__()
        
        
        self.conv1 = nn.Conv2d(inFeatures,outFeatures,kSize[0],stride[0],bias=False)
        self.bn1 = nn.BatchNorm2d(outFeatures)
        self.conv2 = nn.Conv2d(outFeatures,outFeatures,kSize[1],stride[1],padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(outFeatures)
        self.conv3 = nn.Conv2d(outFeatures,outFeatures*4,kSize[2],stride[2],bias=False)
        self.bn3 = nn.BatchNorm2d(outFeatures*4) 
        self.relu = nn.ReLU(True)
        

        self.ds = dn_sample
        if dn_sample:
            self.downSample = nn.Sequential(
                nn.Conv2d(inFeatures,outFeatures*4,1,stride=dnSample_stride,bias=False),
                nn.BatchNorm2d(outFeatures*4)            
            )
        
    
    def forward(self,x):
        x0 = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        if self.ds:
            x0 = self.downSample(x0)
        x = x+x0
        return x


# ResNet
class ResNet(nn.Module):

    def __init__(self,in_channels=3,num_residual_block=[3,4,6,3],num_class=1000,block_type='normal'):
        super(ResNet,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,64,7,2,3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(3,2,1)

        if block_type.lower() == 'bottleneck':    
            self.resnet,outchannels = self.__bottlenecks(num_residual_block)
        else:
            self.resnet,outchannels = self.__layers(num_residual_block)
    

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_features=outchannels,out_features=num_class,bias=True)

        
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.resnet(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x 
    
    def __layers(self,num_residual_block):
        layer=[]
        layer += [BasicBlock()]*num_residual_block[0]
        inchannels=64
        for numOFlayers in num_residual_block[1:]:
            stride = [2,1] #updating the stride, the first layer of residual block
            # will have a stride of two and the 2nd layer of the residual block have 
            # a stride of 1
            downsample=True
            outchannels = inchannels*2
            for _ in range(numOFlayers):
                layer.append(BasicBlock(inchannels,outchannels,stride,down_sample=downsample))
                inchannels = outchannels
                downsample = False 
                stride=[1,1]
            
        return nn.Sequential(*layer),outchannels

    
    def __bottlenecks(self,numres):
        
        layer=[]
        
        stride = [1,1,1]
        dnStride=1
        inchan = 64
        for i,numOFlayers in enumerate(numres):
            dn_sample = True
            outchan = 64*(2**i)

            for _ in range(numOFlayers):
                layer+=[ 
                    Bottleneck(inchan,outchan,stride=stride,
                    dn_sample=dn_sample,dnSample_stride=dnStride)
                ]
                inchan = outchan*4
                dn_sample = False
                stride = [1,1,1]   
            dn_sample=True 
            stride = [1,2,1]
            dnStride=2
            

        return nn.Sequential(*layer),inchan