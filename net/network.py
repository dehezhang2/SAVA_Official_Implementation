import torch
import torch.nn as nn
import torchvision
import net.utils as utils

FEATURE_CHANNEL = 512

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

# Encoder
class Encoder(nn.Module):
    def __init__(self, use_gpu=True, encoder = None):
        super(Encoder, self).__init__()
        encoder = encoder
        if use_gpu == True:
            vgg.cuda()
        enc_layers = list(encoder.children())
        self.layers = [nn.Sequential(*enc_layers[:4]),
                      nn.Sequential(*enc_layers[4:11]),
                      nn.Sequential(*enc_layers[11:18]),
                      nn.Sequential(*enc_layers[18:31]),
                      nn.Sequential(*enc_layers[31:44])]

        self.layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
        
    def forward(self, x):
        out = {}
        for i in range(0, len(self.layers)):
            x = self.layers[i](x)
            out[self.layer_names[i]] = x
        return out
    
    def __str__(self):
        ans = ''
        for layer in self.layers:
            ans += (layer.__str__() + '\n')
        return ans

# Self Attention Module
class SelfAttention(nn.Module):
    def __init__(self, attn = None):
        super(SelfAttention, self).__init__()
        self.f = nn.Conv2d(FEATURE_CHANNEL, FEATURE_CHANNEL // 2, kernel_size=1) # [b, floor(c/2), h, w]
        self.g = nn.Conv2d(FEATURE_CHANNEL, FEATURE_CHANNEL // 2, kernel_size=1) # [b, floor(c/2), h, w]
        self.h = nn.Conv2d(FEATURE_CHANNEL, FEATURE_CHANNEL, kernel_size=1)      # [b, c, h, w]
        self.softmax = nn.Softmax(dim=-1)
        self.attn = attn


    def forward(self, x):
        if self.attn == None:
            x_size = x.shape
            f = utils.hw_flatten(self.f(x)).permute(0, 2, 1) # [b, n, c']
            g = utils.hw_flatten(self.g(x)) # [b, c', n]
            h = utils.hw_flatten(self.h(x)) # [b, c, n]
            energy = torch.bmm(f, g) # [b, n, n]
            attention = self.softmax(energy) # [b, n, n]

            ret = torch.mean(attention.permute(0, 2, 1), 2, keepdim=True).permute(0, 2, 1)
            ret = ret.repeat(1, 512, 1)
            # ret = torch.bmm(h, attention.permute(0, 2, 1)) # [b, c, n]
            ret = ret.view(x_size)# [b, c, h, w]
            return ret
        return self.attn

# Reconstruction Network with Self-attention Module
class AttentionNet(nn.Module):
    def __init__(self, attn, encoder, decoder):
        super(AttentionNet, self).__init__()
        self.perceptual_loss_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']

        self.recons_weight = 10
        self.perceptual_weight = 1
        self.tv_weight = 10
        self.attention_weight = 6

        self.encode = encoder
        self.self_attn = attn
        self.decode = decoder

        self.mse_loss = nn.MSELoss()

    def get_encoder(self):
        return self.encode

    def self_attention_autoencoder(self, x, cal_self_attn, projection_method='AdaIN'): # in case kernels are not seperated
        # input_features = self.encode(x)
        # projected_hidden_feature, colorization_kernels, mean_features = utils.project_features(input_features['conv4'], projection_method)
        # attention_feature_map = cal_self_attn(projected_hidden_feature)
        #
        # hidden_feature = projected_hidden_feature * attention_feature_map + projected_hidden_feature
        # hidden_feature = utils.reconstruct_features(hidden_feature, colorization_kernels, mean_features, projection_method)
        #
        # output = self.decode(hidden_feature, input_features)
        # return output, attention_feature_map
        pass
    def calc_recon_loss(self, x, target):
        recons_loss = self.mse_loss(x, target)
        return recons_loss
    
    def calc_perceptual_loss(self, x, target):
        input_feat = self.encode(x)
        output_feat = self.encode(target)
        
        perceptual_loss = 0.0

        for layer in self.perceptual_loss_layers:
            input_per_feat = input_feat[layer]
            output_per_feat = output_feat[layer]
            perceptual_loss += self.mse_loss(input_per_feat, output_per_feat)
        return perceptual_loss

    def calc_tv_loss(self, x):
        tv_loss = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) 
        tv_loss += torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        return tv_loss
    
    def calc_attention_loss(self, att_map):
        return torch.norm(att_map, p=1)

    def forward(self, x):
        # returns loss, output, attention_map
        # seperate must be False in this case
        output, attention_feature_map = self.self_attention_autoencoder(x, self.self_attn)
        output = utils.batch_mean_image_subtraction(output)
        recon_loss = self.calc_recon_loss(x, output) * (255**2 / 4)
        perceptual_loss =  self.calc_perceptual_loss(x, output) * (255**2 / 4)
        tv_loss = self.calc_tv_loss(output) * (255 / 2)
        attention_loss = self.calc_attention_loss(attention_feature_map)
        total_loss = recon_loss * self.recons_weight + perceptual_loss * self.perceptual_weight \
                    + tv_loss * self.tv_weight + attention_loss * self.attention_weight
        loss_dict = {'total': total_loss, 'construct': recon_loss, 'percept': perceptual_loss, 'tv': tv_loss, 'attn': attention_loss}
        return loss_dict, output, attention_feature_map

    
    # def forward(self, content, style):
    #     pass

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

class SANet(nn.Module):
    def __init__(self, in_planes):
        super(SANet, self).__init__()
        self.f = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.g = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim = -1)
        self.out_conv = nn.Conv2d(in_planes, in_planes, (1, 1))
    def forward(self, content, style):
        F = self.f(mean_variance_norm(content))
        G = self.g(mean_variance_norm(style))
        H = self.h(style)
        b, c, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        b, c, h, w = G.size()
        G = G.view(b, -1, w * h)
        S = torch.bmm(F, G)
        S = self.sm(S)
        b, c, h, w = H.size()
        H = H.view(b, -1, w * h)
        O = torch.bmm(H, S.permute(0, 2, 1))
        b, c, h, w = content.size()
        O = O.view(b, c, h, w)
        O = self.out_conv(O)
        O += content
        return O

class Transform(nn.Module):
    def __init__(self, in_planes):
        super(Transform, self).__init__()
        self.sanet4_1 = SANet(in_planes = in_planes)
        self.sanet5_1 = SANet(in_planes = in_planes)
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))
    def forward(self, content4_1, style4_1, content5_1, style5_1):
        print(content4_1.shape)
        print(style4_1.shape)
        print(content5_1.shape)
        print(style5_1.shape)
        return self.merge_conv(self.merge_conv_pad(
            self.sanet4_1(content4_1, style4_1) + self.upsample5_1(self.sanet5_1(content5_1, style5_1))))
    
    

