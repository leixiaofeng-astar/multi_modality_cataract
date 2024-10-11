import os
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch
from PIL import Image
from efficientnet_pytorch import EfficientNet


class Dataset(torch.utils.data.Dataset):
    #'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, labels, transform = None):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform

    def __len__(self):
        #'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        #'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        try:
            # Load data and get label
            X = Image.open(ID)
            ID = os.path.basename(ID)
            # ID = os.path.splitext(fileName)[0]
            y = self.labels[ID]
        except:
            print(ID)
            import pdb
            pdb.set_trace()

        if self.transform:
            X = self.transform(X)

        return X, y

def get_model(model_name, num_class):
    # 'VGG16', 'VGG19bn', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
    #                    'DenseNet121', 'DenseNet169', 'DenseNet201'
    if model_name == 'VGG16':
        model = VGG16(num_class)
        return model
    if model_name == 'VGG19bn':
        model = VGG19bn(num_class)
        return model
    if model_name == 'ResNet18':
        model = ResNet18(num_class)
        return model
    if model_name == 'ResNet34':
        model = ResNet34(num_class)
        return model
    if model_name == 'ResNet50':
        model = ResNet50(num_class)
        return model
    if model_name == 'ResNet101':
        model = ResNet101(num_class)
        return model
    if model_name == 'ResNet152':
        model = ResNet152(num_class)
        return model
    if model_name == 'DenseNet121':
        model = DenseNet121(num_class)
        return model
    if model_name == 'DenseNet169':
        model = DenseNet169(num_class)
        return model
    if model_name == 'DenseNet201':
        model = DenseNet201(num_class)
        return model
    if model_name == 'MNASNet1_0':
        model = MNASNet1_0(num_class)
        return model
    if model_name == 'MobileNet_v2':
        model = MobileNet_v2(num_class)
        return model
    if model_name == 'ResNext50_32x4d':
        model = ResNext50_32x4d(num_class)
        return model
    if model_name == 'Wide_ResNet50_2':
        model = Wide_ResNet50_2(num_class)
        return model
    if model_name == 'ResNext101_32x8d':
        model = ResNext50_32x4d(num_class)
        return model
    if model_name == 'Wide_ResNet101_2':
        model = Wide_ResNet50_2(num_class)
        return model
    if model_name == 'ShuffleNet_v2_x1_0':
        model = ShuffleNet_v2_x1_0(num_class=num_class)
        return model
    if model_name == 'EfficientNet_b0':
        model = EfficientNetb0(num_class=num_class)
        return model
    if model_name == 'EfficientNet_b3':
        model = EfficientNetb3(num_class=num_class)
        return model
    if model_name == 'EfficientNet_b4':
        model = EfficientNetb4(num_class=num_class)
        return model
    if model_name == 'EfficientNet_b6':
        model = EfficientNetb6(num_class=num_class)
        return model
    if model_name == 'EfficientNet_b7':
        model = EfficientNetb7(num_class=num_class)
        return model

class EfficientNetb0(nn.Module):
    def __init__(self, num_class=2):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(EfficientNetb0, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        self.model._fc = nn.Linear(in_features=self.model._fc.in_features, out_features=num_class)

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        bs = x.size(0)
        # Convolution layers
        x = self.model.extract_features(x)

        # Pooling and final linear layer
        x = self.model._avg_pooling(x)
        features = x.view(bs, -1)
        features = self.model._dropout(features)
        out = self.model._fc(features)
        out = torch.softmax(out, dim=1)
        return features, out

class EfficientNetb3(nn.Module):
    def __init__(self, num_class=2):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(EfficientNetb3, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b3')
        self.model._fc = nn.Linear(in_features=self.model._fc.in_features, out_features=num_class)

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        bs = x.size(0)
        # Convolution layers
        x = self.model.extract_features(x)

        # Pooling and final linear layer
        x = self.model._avg_pooling(x)
        features = x.view(bs, -1)
        features = self.model._dropout(features)
        out = self.model._fc(features)
        out = torch.softmax(out, dim=1)
        return features, out

class EfficientNetb4(nn.Module):
    def __init__(self, num_class=2):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(EfficientNetb4, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b4')
        self.model._fc = nn.Linear(in_features=self.model._fc.in_features, out_features=num_class)

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        bs = x.size(0)
        # Convolution layers
        x = self.model.extract_features(x)

        # Pooling and final linear layer
        x = self.model._avg_pooling(x)
        features = x.view(bs, -1)
        features = self.model._dropout(features)
        out = self.model._fc(features)
        out = torch.softmax(out, dim=1)
        return features, out

class EfficientNetb6(nn.Module):
    def __init__(self, num_class=2):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(EfficientNetb6, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b6')
        self.model._fc = nn.Linear(in_features=self.model._fc.in_features, out_features=num_class)

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        bs = x.size(0)
        # Convolution layers
        x = self.model.extract_features(x)

        # Pooling and final linear layer
        x = self.model._avg_pooling(x)
        features = x.view(bs, -1)
        features = self.model._dropout(features)
        out = self.model._fc(features)
        out = torch.softmax(out, dim=1)
        return features, out

class EfficientNetb7(nn.Module):
    def __init__(self, num_class=2):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(EfficientNetb7, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b7')
        self.model._fc = nn.Linear(in_features=self.model._fc.in_features, out_features=num_class)

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        bs = x.size(0)
        # Convolution layers
        x = self.model.extract_features(x)

        # Pooling and final linear layer
        x = self.model._avg_pooling(x)
        features = x.view(bs, -1)
        features = self.model._dropout(features)
        out = self.model._fc(features)
        out = torch.softmax(out, dim=1)
        return features, out

class VGG16(nn.Module):
    def __init__(self, num_class=2):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGG16, self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.vgg_features = vgg.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc_features = nn.Sequential(*list(vgg.classifier.children())[:-2])
        self.fc = nn.Linear(in_features=4096, out_features=num_class)

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = self.vgg_features(x)
        features = self.avgpool(features).view(x.shape[0], -1)
        features = self.fc_features(features).view(x.shape[0], -1)
        # out = torch.softmax(self.fc(features), dim=1)
        out = self.fc(features)
        return features, out

class VGG19bn(nn.Module):
    def __init__(self, num_class=2):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGG19bn, self).__init__()
        vgg = models.vgg19_bn(pretrained=True)
        self.vgg_features = vgg.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc_features = nn.Sequential(*list(vgg.classifier.children())[:-2])
        self.fc = nn.Linear(in_features=4096, out_features=num_class)

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = self.vgg_features(x)
        features = self.avgpool(features).view(x.shape[0], -1)
        features = self.fc_features(features).view(x.shape[0], -1)
        # out = torch.softmax(self.fc(features), dim=1)
        out = self.fc(features)
        return features, out

class ResNet18(nn.Module):
    def __init__(self, num_class=2):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(ResNet18, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=num_class)

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = self.features(x).view(x.shape[0], -1)
        out = torch.softmax(self.fc(features), dim=1)
        return features, out

class ResNet34(nn.Module):
    def __init__(self, num_class=2):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(ResNet34, self).__init__()
        resnet = models.resnet34(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=num_class)

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = self.features(x).view(x.shape[0], -1)
        out = torch.softmax(self.fc(features), dim=1)
        # out = self.fc(features)
        return features, out

class ResNet50(nn.Module):
    def __init__(self, num_class=2):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(ResNet50, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=num_class)

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = self.features(x).view(x.shape[0], -1)
        out = torch.softmax(self.fc(features), dim=1)
        return features, out

class ResNet101(nn.Module):
    def __init__(self, num_class=2):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(ResNet101, self).__init__()
        resnet = models.resnet101(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=num_class)

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = self.features(x).view(x.shape[0], -1)
        out = torch.softmax(self.fc(features), dim=1)
        return features, out

class ResNet152(nn.Module):
    def __init__(self, num_class=2):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(ResNet152, self).__init__()
        resnet = models.resnet152(pretrained=True)
        # num_hidden = 128
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=num_class)

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = self.features(x).view(x.shape[0], -1)
        out = torch.softmax(self.fc(features), dim=1)
        # out = self.fc2(out)
        return features, out

class DenseNet121(nn.Module):
    def __init__(self, num_class=2):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(DenseNet121, self).__init__()
        dense_net = models.densenet121(pretrained=True)
        self.features = dense_net.features
        self.fc = nn.Linear(in_features=dense_net.classifier.in_features, out_features=num_class)

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = self.features(x)
        features = F.relu(features, inplace=True)
        features = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
        # avg_pool_layer = nn.AvgPool2d(7, stride=1)
        # features = avg_pool_layer(features).view(x.shape[0], -1)
        out = torch.softmax(self.fc(features), dim=1)
        return features, out

class DenseNet169(nn.Module):
    def __init__(self, num_class=2):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(DenseNet169, self).__init__()
        dense_net = models.densenet169(pretrained=True)
        self.features = dense_net.features
        self.fc = nn.Linear(
            in_features=dense_net.classifier.in_features, out_features=num_class)

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = self.features(x)
        features = F.relu(features, inplace=True)
        features = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
        out = torch.softmax(self.fc(features), dim=1)
        return features, out

class DenseNet201(nn.Module):
    def __init__(self, num_class=2):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(DenseNet201, self).__init__()
        dense_net = models.densenet201(pretrained=True)
        self.features = dense_net.features
        self.fc = nn.Linear(
            in_features=dense_net.classifier.in_features, out_features=num_class)

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = self.features(x)
        features = F.relu(features, inplace=True)
        features = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
        out = torch.softmax(self.fc(features), dim=1)
        return features, out

class MNASNet1_0(nn.Module):
    def __init__(self, num_class=2):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(MNASNet1_0, self).__init__()
        mnasnet1_0 = models.mnasnet1_0(pretrained=True)
        self.layers = mnasnet1_0.layers
        self.fc = nn.Sequential(nn.Dropout(p=0.2, inplace=True), nn.Linear(
            in_features=1280, out_features=num_class))

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = self.layers(x)
        features = features.mean([2, 3])
        out = torch.softmax(self.fc(features), dim=1)
        return features, out

class MobileNet_v2(nn.Module):
    def __init__(self, num_class=2):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(MobileNet_v2, self).__init__()
        mobileNet_v2 = models.mobilenet_v2(pretrained=True)
        self.features = mobileNet_v2.features
        self.fc = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(
            in_features=mobileNet_v2.last_channel, out_features=num_class))

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = self.features(x)
        features = nn.functional.adaptive_avg_pool2d(features, 1).reshape(features.shape[0], -1)
        out = torch.softmax(self.fc(features), dim=1)
        return features, out

class ResNext50_32x4d(nn.Module):
    def __init__(self, num_class=2):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(ResNext50_32x4d, self).__init__()
        resnet = models.resnext50_32x4d(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=num_class)

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = self.features(x).view(x.shape[0], -1)
        out = torch.softmax(self.fc(features), dim=1)
        # out = self.fc2(out)
        return features, out

class ResNext101_32x8d(nn.Module):
    def __init__(self, num_class=2):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(ResNext101_32x8d, self).__init__()
        resnet = models.resnext101_32x8d(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=num_class)

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = self.features(x).view(x.shape[0], -1)
        out = torch.softmax(self.fc(features), dim=1)
        # out = self.fc2(out)
        return features, out

class Wide_ResNet50_2(nn.Module):
    def __init__(self, num_class=2):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(Wide_ResNet50_2, self).__init__()
        resnet = models.wide_resnet50_2(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=num_class)

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = self.features(x).view(x.shape[0], -1)
        out = torch.softmax(self.fc(features), dim=1)
        # out = self.fc2(out)
        return features, out

class Wide_ResNet101_2(nn.Module):
    def __init__(self, num_class=2):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(Wide_ResNet101_2, self).__init__()
        resnet = models.wide_resnet101_2(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=num_class)

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = self.features(x).view(x.shape[0], -1)
        out = torch.softmax(self.fc(features), dim=1)
        # out = self.fc2(out)
        return features, out

class ShuffleNet_v2_x1_0(nn.Module):
    def __init__(self, num_class=2):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(ShuffleNet_v2_x1_0, self).__init__()
        shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
        self.conv1 = shufflenet.conv1
        self.maxpool = shufflenet.maxpool
        self.stage2 = shufflenet.stage2
        self.stage3 = shufflenet.stage3
        self.stage4 =shufflenet.stage4
        self.conv5 = shufflenet.conv5
        self.fc = nn.Linear(in_features=shufflenet.fc.in_features, out_features=num_class)

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        features = x.mean([2, 3])  # globalpool
        out = torch.softmax(self.fc(features), dim=1)
        # out = self.fc2(out)
        return features, out