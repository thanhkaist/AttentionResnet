import torch.nn as nn
import math
import torch
import torch.nn.functional as F
import pdb

# Utility function

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Norm(nn.Module):
    def __init__(self, name, n_feats):
        super(Norm, self).__init__()
        assert name in ['bn', 'gn', 'gbn', 'none']
        if name == 'bn':
            self.norm = nn.BatchNorm2d(n_feats)
        elif name == 'gn':
            self.norm = nn.GroupNorm(32, n_feats)
        elif name == 'gbn':
            self.norm = nn.Sequential(nn.GroupNorm(32, n_feats, affine=False),nn.BatchNorm2d(n_feats))
        elif name == 'none':
            pass
        self.name = name

    def forward(self, x):
        if self.name == 'none':
            return x
        else:
            return self.norm(x)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class BamSpatialAttention(nn.Module):
    def __init__(self,channel,reduction = 16, dilation_ratio =2):
        super(BamSpatialAttention,self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1),

            nn.BatchNorm2d(channel//reduction),
            nn.Conv2d(channel//reduction,channel//reduction,3,padding=dilation_ratio,dilation=dilation_ratio),
            nn.BatchNorm2d(channel // reduction),
            nn.ReLU(True),

            nn.BatchNorm2d(channel // reduction),
            nn.Conv2d(channel // reduction, channel // reduction, 3, padding=dilation_ratio, dilation=dilation_ratio),
            nn.BatchNorm2d(channel // reduction),
            nn.ReLU(True),

            nn.Conv2d(channel//reduction,1,1)
        )
    def forward(self, x):
        return self.body(x).expand_as(x)


class BamChannelAttention(nn.Module):
    def __init__(self,channel,reduction = 16):
        super(BamChannelAttention,self).__init__()
        self.avgPool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//reduction,channel,1),
        )
    def forward(self,x):
        out = self.avgPool(x)
        out = self.fc(out)
        return out.expand_as(x)




class CBamSpatialAttention(nn.Module):
    def __init__(self):
        super(CBamSpatialAttention,self).__init__()

class CBamChannelAttention(nn.Module):
    def __init__(self):
        super(CBamChannelAttention,self).__init__()



# Attention module
class SE_Attention_Layer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE_Attention_Layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1)
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        y = F.sigmoid(y)
        return  x*y.expand_as(x)

class BAM_Attention_Layer(nn.Module):
    def __init__(self, channel, att = 'both', reduction=16):
        super(BAM_Attention_Layer, self).__init__()
        self.att = att
        self.channelAtt =None
        self.spatialAtt =None
        if att == 'both' or att == 'c':
            self.channelAtt = BamChannelAttention(channel,reduction)
        if att == 'both' or att == 's':
            self.spatialAtt = BamSpatialAttention(channel,reduction)

    def forward(self, x):
        if self.att =='both':
            y1 = self.spatialAtt(x)
            y2 = self.channelAtt(x)
            y = y1* y2
        elif self.att =='c':
            y = self.channelAtt(x)
        elif self.att =='s':
            y = self.spatialAtt(x)
        return (1 +F.sigmoid(y))*x

class CBAM_Attention_Layer(nn.Module):
    def __init__(self, channel):
        super(CBAM_Attention_Layer, self).__init__()
        self.channel_att = SE_Attention_Layer(channel)
        self.spatial_att = BAM_Attention_Layer(channel)

    def forward(self, x):
        y1 = self.channel_att(x)
        y2 = self.spatial_att(x)
        return y1 + y2

# Blocks

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, attention='no', base_width= 64, t_norm = 'bn'):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.sigmoid = nn.Sigmoid()
        if attention == 'se':
            self.att = SE_Attention_Layer(planes * 4)
        elif attention == 'c_bam':
            self.att = BAM_Attention_Layer(planes * 4,'c')
        elif attention == 's_bam':
            self.att = BAM_Attention_Layer(planes * 4,'s')
        elif attention == 'j_bam':
            self.att = BAM_Attention_Layer(planes * 4,'both')
        elif attention == 'c_cbam':
            self.att = CBAM_Attention_Layer(planes * 4,'c')
        elif attention == 's_cbam':
            self.att = CBAM_Attention_Layer(planes * 4,'s')
        elif attention == 'j_cbam':
            self.att = CBAM_Attention_Layer(planes * 4,'both')
        elif attention == 'no':
            self.att = None
        else:
            raise Exception('Unknown att type')

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.att is not None:
            out = self.att(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None, attention ='no'):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if attention == 'se':
            self.att = SE_Attention_Layer(planes, reduction=4)
        elif attention == 'c_bam':
            self.att = BAM_Attention_Layer(planes, 'c')
        elif attention == 's_bam':
            self.att = BAM_Attention_Layer(planes, 's')
        elif attention == 'j_bam':
            self.att = BAM_Attention_Layer(planes, 'both')
        elif attention == 'c_cbam':
            self.att = CBAM_Attention_Layer(planes,'c')
        elif attention == 's_cbam':
            self.att = CBAM_Attention_Layer(planes,'s')
        elif attention == 'j_cbam':
            self.att = CBAM_Attention_Layer(planes,'both')

        elif attention == 'no':
            self.att = None
        else:
            raise Exception('Unknown attention type')

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.att is not None:
            att = self.att(out)
            att = self.sigmoid(att)
            out = out*att

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, norm='bn', attention='no', num_classes=100):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.size = 64
        self.norm = norm


        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, attention=attention)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1, attention=attention)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, attention=attention)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, attention=attention)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, attention='no'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
                Norm(self.norm, planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, base_width=self.size, t_norm=self.norm, attention=attention))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, base_width=self.size, t_norm=self.norm, attention=attention))
        # append att layer to the stage
        #layers.append(block(self.inplanes, planes, attention=attention, size=size))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def se_resnet50(attention="channel",norm = 'bn',**kwargs):
    if attention == "channel":
        model = ResNet(Bottleneck, [3, 4, 6, 3],norm, 'se', **kwargs)
    else:
        raise Exception("SEnet only support channel attention")
    return model


def bam_resnet50(attention="joint",norm = 'bn',**kwargs):
    if attention == "channel":
        model = ResNet(Bottleneck, [3, 4, 6, 3], norm, 'c_bam', **kwargs)
    elif attention == "spatial":
        model = ResNet(Bottleneck, [3, 4, 6, 3], norm, 's_bam', **kwargs)
    elif attention == "joint":
        model = ResNet(Bottleneck, [3, 4, 6, 3], norm, 'j_bam', **kwargs)
    else:
        raise Exception("Unknown attention for BAM")
    return model

def cbam_resnet50(attention="joint",norm = 'bn',**kwargs):
    if attention == "channel":
        model = ResNet(Bottleneck, [3, 4, 6, 3], norm, 'c_cbam', **kwargs)
    elif attention == "spatial":
        model = ResNet(Bottleneck, [3, 4, 6, 3], norm, 's_cbam', **kwargs)
    elif attention == "joint":
        model = ResNet(Bottleneck, [3, 4, 6, 3], norm, 'j_cbam', **kwargs)
    else:
        raise Exception("Unknown attention for CBAM")
    return model


def resnet50(attention="no",norm = 'bn',**kwargs):
    if attention == "no":
        model = ResNet(Bottleneck, [3, 4, 6, 3], norm,'no', **kwargs)
    else:
        raise Exception("Unknown attention for baseline resnet")
    return model




