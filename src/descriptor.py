from torchvision.models import vgg19
import torch as th


class VGG19Loss(th.nn.Module):

    def __init__(self, device):
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

        for i in [4, 9, 18, 27, 36]: # without BN
            self.net[i].register_forward_hook(hook)

        # weight proportional to num. of feature channels [Aittala 2016]
        self.weights = [1, 2, 4, 8, 8]


    def gram(self, x):
        self.outputs = []

        # run VGG features
        x = self.net(x)

        self.outputs.append(x)
        self.outputs.pop()

        result = []
        for i, feature in enumerate(self.outputs):
            n, f, s1, s2 = feature.shape
            s = s1 * s2
            feature = feature.view((n*f, s))

            # Gram matrix
            G = th.mm(feature, feature.t()) / s
            result.append(G.flatten() * self.weights[i])

        return th.cat(result)

    def load(self, im):
        self.im_gram = self.gram(im)

    def forward(self, im):
        return self.criterion(self.gram(im), self.im_gram)