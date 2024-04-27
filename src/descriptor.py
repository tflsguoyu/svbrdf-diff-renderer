import torch as th
from torchvision.models import vgg19
from torchvision.transforms import Normalize


class VGG19Loss(th.nn.Module):

    def __init__(self, device, weight="1111"):
        super(VGG19Loss, self).__init__()
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

        for i in [1, 3, 13, 22]: # r11, r12, r32, r42
            self.net[i].register_forward_hook(hook)
        if weight == "1148":
            self.weights = [0.071, 0.071, 0.286, 0.572]
        elif weight == "8821":
            self.weights = [0.421, 0.421, 0.105, 0.053]
        else:
            self.weights = [0.25, 0.25, 0.25, 0.25]


        # for i in [4, 9, 18, 27, 36]: # r12, r22, r34, r44, r54
        #     self.net[i].register_forward_hook(hook)
        # # weight proportional to num. of feature channels [Aittala 2016]
        # self.weights = [1, 2, 4, 8, 8]

        # for i in [3, 8, 15, 24]: # r12, r22, r33, r43
        #     self.net[i].register_forward_hook(hook)
        # self.weights = [1, 2, 4, 8]


    def gram(self, x, is_gram=False):
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
        self.im_gram = self.gram(self.normalize(im))

    def normalize(self, im):
        # im: N x C x H x W
        transform = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.255]
        )
        im_norm = im.clone()
        for i in range(im.shape[0]):
            im_norm[i,:,:,:] = transform(im[i,:,:,:])

        return im_norm

    def forward(self, im):
        return self.criterion(self.gram(self.normalize(im)), self.im_gram)
