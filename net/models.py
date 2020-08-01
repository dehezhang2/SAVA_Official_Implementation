from net.network import AttentionNet, Correlation
import torch
import torch.nn as nn
import torch.nn.functional as F
import net.utils as utils
from net.utils import truncated_normal_, KMeans, project_features

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

class Transform(nn.Module):
    def __init__(self, in_planes, self_attn):
        super(Transform, self).__init__()
        self.corr4_1 = Correlation(in_planes=in_planes, hidden_planes=in_planes)
        self.corr5_1 = Correlation(in_planes=in_planes, hidden_planes=in_planes)
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')

        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))

        self.self_attn = self_attn

    def

    def forward(self, content4_1, style4_1, content5_1, style5_1):
        feature_corr4_1 = self.corr4_1(content4_1, style4_1)
        feature_corr5_1 = self.corr5_1(content5_1, style5_1)

        _, content_attn4_1 = self.self_attn(content4_1)
        _, style_attn4_1 = self.self_attn(style4_1)
        _, content_attn5_1 = self.self_attn(content5_1)
        _, style_attn5_1 = self.self_attn(style5_1)


        return self.merge_conv(self.merge_conv_pad(
            self.sanet4_1(content4_1, style4_1) + self.upsample5_1(self.sanet5_1(content5_1, style5_1))))

class SAVA_test:
    def __init__(self, attention_net, transformer):
        self.attention_net = attention_net
        self.encode = attention_net.get_encoder()
        self.transformer = transformer

    def attention_filter(self, attention_feature_map, kernel_size=3, mean=6, stddev=5):
        attention_map = torch.abs(attention_feature_map)

        attention_mask = attention_map > 2 * torch.mean(attention_map)
        attention_mask = attention_mask.float()

        w = torch.randn(kernel_size, kernel_size)
        truncated_normal_(w, mean, stddev)
        w = w / torch.sum(w)

        # [in_channels, out_channels, filter_height, filter_width]
        w = torch.unsqueeze(w, 0)
        w = w.repeat(attention_mask.shape[1], 1, 1)

        w = torch.unsqueeze(w, 0)
        w = w.repeat(attention_mask.shape[1], 1, 1, 1)

        gaussian_filter = nn.Conv2d(attention_mask.shape[1], attention_mask.shape[1], (kernel_size, kernel_size))
        gaussian_filter.weight.data = w
        gaussian_filter.weight.requires_grad = False
        pad_filter = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            gaussian_filter
        )
        pad_filter.cuda()
        attention_map = pad_filter(attention_mask)
        attention_map = attention_map - torch.min(attention_map)
        attention_map = attention_map / torch.max(attention_map)
        return attention_map

    def transfer(self, contents, styles, inter_weight=1, projection_method='ZCA'):
        content_features = self.attention_net.encode(contents)
        style_features = self.attention_net.encode(styles)

        content_hidden_feature_4 = content_features[self.attention_net.perceptual_loss_layers[-2]]
        style_hidden_feature_4 = style_features[self.attention_net.perceptual_loss_layers[-2]]
        content_hidden_feature_5 = content_features[self.attention_net.perceptual_loss_layers[-1]]
        style_hidden_feature_5 = style_features[self.attention_net.perceptual_loss_layers[-1]]

        projected_content_features, content_kernels, mean_content_features = utils.project_features(
            content_hidden_feature_4, projection_method)
        projected_style_features, style_kernels, mean_style_features = utils.project_features(style_hidden_feature_4,
                                                                                              projection_method)

        content_attention_map = self.attention_net.self_attn(projected_content_features)
        style_attention_map = self.attention_net.self_attn(projected_style_features)

        # projected_content_features = projected_content_features * content_attention_feature_map + projected_content_features
        content_attention_map = self.attention_filter(content_attention_map)
        style_attention_map = self.attention_filter(style_attention_map)

        swapped_features = self.transformer(content_hidden_feature_4, style_hidden_feature_4, content_hidden_feature_5, style_hidden_feature_5)

        output = self.attention_net.decode(swapped_features)
        output = utils.batch_mean_image_subtraction(output)

        return output, content_attention_map, style_attention_map
