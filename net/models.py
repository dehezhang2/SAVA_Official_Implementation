from net.network import AttentionNet, Transform
import torch
import torch.nn as nn
import torch.nn.functional as F
import net.utils as utils
from net.utils import truncated_normal_, KMeans, project_features

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
