import torch.nn as nn
import torch.nn.functional as F


def get_mean_and_std(feat):
    N, C, H, W = feat.size()
    feat_mean = feat.reshape(N, C, H * W).mean(dim=2).reshape(N, C, 1, 1)
    feat_std = feat.reshape(N, C, H * W).std(dim=2).reshape(N, C, 1, 1)
    return feat_mean, feat_std


def AdaIN(content_feats, style_feats, epsilon=1e-6):
    content_mean, content_std = get_mean_and_std(content_feats)
    style_mean, style_std = get_mean_and_std(style_feats)
    norm_feat = (content_feats - content_mean) / (content_std + epsilon)
    denorm_feat = norm_feat * style_std + style_mean
    return denorm_feat


def content_loss(output_feats, content_target):
    return F.mse_loss(output_feats, content_target)


def style_loss(output_intermediate, style_intermediate):
    assert len(output_intermediate) == len(style_intermediate)
    style_loss = 0
    for output_feats, style_feats in zip(output_intermediate, style_intermediate):
        output_mean, output_std = get_mean_and_std(output_feats)
        style_mean, style_std = get_mean_and_std(style_feats)
        style_loss += F.mse_loss(output_mean, style_mean) + F.mse_loss(output_std, style_std)
    return style_loss
