import numpy as np
import torch as th
from torchvision.models import vgg19
from torchvision.transforms import Normalize


class VGGLoss(th.nn.Module):
    def __init__(self, device, weights=np.array([1, 1, 1, 1])/4):
        super(VGGLoss, self).__init__()
        self.criterion = th.nn.MSELoss().to(device)

        # get VGG19 feature network in evaluation mode
        self.net = vgg19(True).features.to(device)
        self.net.eval()

        # change max pooling to average pooling
        for i, x in enumerate(self.net):
            if isinstance(x, th.nn.MaxPool2d):
                self.net[i] = th.nn.AvgPool2d(kernel_size=2)

        def hook(module, input, output):
            self.outputs.append(output)

        # print(self.net)

        for i in [1, 3, 13, 22]:  # r11, r12, r32, r42
            self.net[i].register_forward_hook(hook)
        self.weights = weights

        # for i in [4, 9, 18, 27, 36]:  # r12, r22, r34, r44, r54
        #     self.net[i].register_forward_hook(hook)
        # # weight proportional to num. of feature channels [Aittala 2016]
        # self.weights = [1, 2, 4, 8, 8]

        # for i in [3, 8, 15, 24]:  # r12, r22, r33, r43
        #     self.net[i].register_forward_hook(hook)
        # self.weights = [1, 2, 4, 8]

    def compute_feature_vector(self, x, is_gram=False):
        self.outputs = []

        # run VGG features
        x = self.net(x)

        self.outputs.append(x)
        self.outputs.pop()

        result = []
        for i, feature in enumerate(self.outputs):
            if is_gram:
                n, f, s1, s2 = feature.shape
                s = s1 * s2
                feature = feature.view((n*f, s))
                # Gram matrix
                G = th.mm(feature, feature.t()) / s
                result.append(G.flatten() * self.weights[i])
            else:
                result.append(feature.flatten() * self.weights[i])

        return th.cat(result)

    def load(self, im):
        self.im_feature = self.compute_feature_vector(self.normalize(im))

    def normalize(self, im):
        # im: N x C x H x W
        transform = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.255]
        )
        im_norm = im.clone()
        for i in range(im.shape[0]):
            im_norm[i, :, :, :] = transform(im[i, :, :, :])

        return im_norm

    def forward(self, x):
        x_feature = self.compute_feature_vector(self.normalize(x))
        return self.criterion(x_feature, self.im_feature)


# ------------------ Used in old MaterialGAN -----------------------
# import torch as th
# import torch.nn.functional as F
# from torchvision.transforms import Normalize


# class VGGLoss(th.nn.Module):
#     def __init__(self, device, weights, ckp="ckp/vgg_conv.pt"):
#         super(VGGLoss, self).__init__()

#         self.net = VGG()
#         self.net.load_state_dict(th.load(ckp))
#         self.net.eval().to(device)

#         self.layer = ['r11','r12','r32','r42']
#         self.weights = weights

#         self.criterion = th.nn.MSELoss().to(device)

#     def forward(self, x):
#         x_feature = self.compute_feature_vector(self.normalize(x))
#         return self.criterion(x_feature, self.im_feature)

#     def load(self, im):
#         self.im_feature = self.compute_feature_vector(self.normalize(im))

#     def compute_feature_vector(self, x):
#         outputs = self.net(x, self.layer)
#         result = []
#         for i, feature in enumerate(outputs):
#             result.append(feature.flatten() * self.weights[i])
#         return th.cat(result)

#     def normalize(self, im):
#         # im: N x C x H x W
#         transform = Normalize(
#             mean=[0.48501961, 0.45795686, 0.40760392],
#             std=[1./255, 1./255, 1./255]
#         )
#         im_norm = im.clone()
#         for i in range(im.shape[0]):
#             im_norm[i,:,:,:] = transform(im[i,:,:,:])
#         return im_norm


# class VGG(th.nn.Module):
#     def __init__(self, pool='max'):
#         super(VGG, self).__init__()
#         #vgg modules
#         self.conv1_1 = th.nn.Conv2d(3, 64, kernel_size=3, padding=1)
#         self.conv1_2 = th.nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.conv2_1 = th.nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.conv2_2 = th.nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         self.conv3_1 = th.nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.conv3_2 = th.nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.conv3_3 = th.nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.conv3_4 = th.nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.conv4_1 = th.nn.Conv2d(256, 512, kernel_size=3, padding=1)
#         self.conv4_2 = th.nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.conv4_3 = th.nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.conv4_4 = th.nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.conv5_1 = th.nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.conv5_2 = th.nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.conv5_3 = th.nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.conv5_4 = th.nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         if pool == 'max':
#             self.pool1 = th.nn.MaxPool2d(kernel_size=2, stride=2)
#             self.pool2 = th.nn.MaxPool2d(kernel_size=2, stride=2)
#             self.pool3 = th.nn.MaxPool2d(kernel_size=2, stride=2)
#             self.pool4 = th.nn.MaxPool2d(kernel_size=2, stride=2)
#             self.pool5 = th.nn.MaxPool2d(kernel_size=2, stride=2)
#         elif pool == 'avg':
#             self.pool1 = th.nn.AvgPool2d(kernel_size=2, stride=2)
#             self.pool2 = th.nn.AvgPool2d(kernel_size=2, stride=2)
#             self.pool3 = th.nn.AvgPool2d(kernel_size=2, stride=2)
#             self.pool4 = th.nn.AvgPool2d(kernel_size=2, stride=2)
#             self.pool5 = th.nn.AvgPool2d(kernel_size=2, stride=2)

#     def forward(self, x, out_keys):
#         out = {}
#         out['r11'] = F.relu(self.conv1_1(x))
#         out['r12'] = F.relu(self.conv1_2(out['r11']))
#         out['p1'] = self.pool1(out['r12'])
#         out['r21'] = F.relu(self.conv2_1(out['p1']))
#         out['r22'] = F.relu(self.conv2_2(out['r21']))
#         out['p2'] = self.pool2(out['r22'])
#         out['r31'] = F.relu(self.conv3_1(out['p2']))
#         out['r32'] = F.relu(self.conv3_2(out['r31']))
#         out['r33'] = F.relu(self.conv3_3(out['r32']))
#         out['r34'] = F.relu(self.conv3_4(out['r33']))
#         out['p3'] = self.pool3(out['r34'])
#         out['r41'] = F.relu(self.conv4_1(out['p3']))
#         out['r42'] = F.relu(self.conv4_2(out['r41']))
#         out['r43'] = F.relu(self.conv4_3(out['r42']))
#         # out['r44'] = F.relu(self.conv4_4(out['r43']))
#         # out['p4'] = self.pool4(out['r44'])
#         # out['r51'] = F.relu(self.conv5_1(out['p4']))
#         # out['r52'] = F.relu(self.conv5_2(out['r51']))
#         # out['r53'] = F.relu(self.conv5_3(out['r52']))
#         # out['r54'] = F.relu(self.conv5_4(out['r53']))
#         # out['p5'] = self.pool5(out['r54'])
#         return [out[key] for key in out_keys]
