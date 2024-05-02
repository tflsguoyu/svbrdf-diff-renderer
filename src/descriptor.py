import os
import torch as th
import torch.nn.functional as F
from torchvision.transforms import Normalize


class VGGLoss(th.nn.Module):
    def __init__(self, device, weights, ckp="ckp/vgg_conv.pt"):
        super(VGGLoss, self).__init__()

        self.net = VGG()
        self.net.load_state_dict(th.load(ckp))
        self.net.eval().to(device)

        self.layer = ['r11','r12','r32','r42']
        self.weights = weights

        self.criterion = th.nn.MSELoss().to(device)

    def forward(self, x):
        x_feature = self.compute_feature_vector(self.normalize(x))
        return self.criterion(x_feature, self.im_feature)

    def load(self, im):
        self.im_feature = self.compute_feature_vector(self.normalize(im))

    def compute_feature_vector(self, x):
        outputs = self.net(x, self.layer)
        result = []
        for i, feature in enumerate(outputs):
            result.append(feature.flatten() * self.weights[i])
        return th.cat(result)

    def normalize(self, im):
        # im: N x C x H x W
        transform = Normalize(
            mean=[0.48501961, 0.45795686, 0.40760392],
            std=[1./255, 1./255, 1./255]
        )
        im_norm = im.clone()
        for i in range(im.shape[0]):
            im_norm[i,:,:,:] = transform(im[i,:,:,:])
        return im_norm


class VGG(th.nn.Module):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        #vgg modules
        self.conv1_1 = th.nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = th.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = th.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = th.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = th.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = th.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = th.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = th.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = th.nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = th.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = th.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = th.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = th.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = th.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = th.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = th.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = th.nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = th.nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = th.nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = th.nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = th.nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = th.nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = th.nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = th.nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = th.nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = th.nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        # out['r44'] = F.relu(self.conv4_4(out['r43']))
        # out['p4'] = self.pool4(out['r44'])
        # out['r51'] = F.relu(self.conv5_1(out['p4']))
        # out['r52'] = F.relu(self.conv5_2(out['r51']))
        # out['r53'] = F.relu(self.conv5_3(out['r52']))
        # out['r54'] = F.relu(self.conv5_4(out['r53']))
        # out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]
