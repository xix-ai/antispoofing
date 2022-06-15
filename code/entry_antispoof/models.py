from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ModelEMA(nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super().__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def forward(self, input):
        return self.module(input)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.parameters(), model.parameters()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))
            for ema_v, model_v in zip(self.module.buffers(), model.buffers()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(model_v)

    def update_parameters(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        self.module.load_state_dict(state_dict)


class AttrModelOld(nn.Module):
    def __init__(
            self,
            encoder,
            num_classes=[1, 1, 3, 1, 1],
            pretrained=False,
            drop_rate=0.0,
            drop_path_rate=0.0,
            num_feat=1536
    ):
        super().__init__()
        self.model = timm.create_model(
            encoder,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        attrs = []
        for cls in num_classes:
            attrs.append(nn.Linear(num_feat, cls))
        self.attrs = nn.ModuleList(attrs)

    def forward(self, x):
        feat = self.model(x)

        res = []
        for clsassif in self.attrs:
            res.append(clsassif(feat))

        return torch.stack(res)


class AttrLinear(nn.Module):
    def __init__(
            self,
            encoder,
            num_classes=[1, 1, 3, 1, 1],
            drop_rate=0.0,
            weights=None,
            # drop_path_rate=0.0,
            num_feat=1536
    ):
        super().__init__()
        pretrained = True if weights is None else False
        self.model = timm.create_model(
            encoder,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate,
        )
        if weights is not None:
            st = torch.load(weights)
            self.model.load_state_dict(st, strict=True)
        self.model.eval()
        attrs = []
        for cls in num_classes:
            attrs.append(nn.Linear(num_feat, cls))
        self.attrs = nn.ModuleList(attrs)

    def forward(self, x):
        with torch.no_grad():
            feat = self.model(x)

        res = []
        for clsassif in self.attrs:
            res.append(clsassif(feat))

        return torch.stack(res)


class AttrModel(nn.Module):
    def __init__(
            self,
            encoder,
            num_attrs=7,
            pretrained=False,
            drop_rate=0.0,
            drop_path_rate=0.0,
    ):
        super().__init__()
        self.model = timm.create_model(
            encoder,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        num_feat = self.model.get_classifier().in_features
        self.attrs = nn.Linear(num_feat, num_attrs)

    def forward(self, x):
        feat = self.model(x)

        return self.attrs(feat)


class SpatialAttention(nn.Module):
    def __init__(self, kernel=3):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(
            2, 1, kernel_size=kernel, padding=kernel // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        return self.sigmoid(x)


class FeatModel(nn.Module):
    def __init__(
            self,
            encoder,
            num_classes=1,
            pretrained=False,
            drop_rate=0.0,
            drop_path_rate=0.0,
    ):
        super().__init__()
        self.bn = nn.BatchNorm2d(3)
        self.model = timm.create_model(
            encoder,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            features_only=True,
        )
        self.drop_rate = drop_rate
        self.sa1 = SpatialAttention(kernel=7)
        self.sa2 = SpatialAttention(kernel=5)
        self.sa3 = SpatialAttention(kernel=3)
        # self.sa4 = SpatialAttention(kernel=3)
        self.downsample32x32 = nn.Upsample(size=(32, 32), mode="bilinear")
        num_feat_concat = 0
        for n in range(1, 4):
            num_feat_concat += self.model.feature_info.info[n]["num_chs"]
        self.lastconv1 = nn.Sequential(
            nn.Conv2d(
                num_feat_concat, 160, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(160),
            nn.ReLU(),
            nn.Conv2d(160, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )

        num_feat = self.model.feature_info.info[-1]["num_chs"]
        num_feat_head = 1280
        self.conv_head = nn.Conv2d(
            num_feat, num_feat_head, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_feat_head)
        self.act2 = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(num_feat_head, num_classes)

    def forward(self, x):
        # x = self.bn(x)
        feats = self.model(x)
        attention1 = self.sa1(feats[1])
        x_Block1_SA = attention1 * feats[1]
        x_Block1_32x32 = self.downsample32x32(x_Block1_SA)

        attention2 = self.sa2(feats[2])
        x_Block2_SA = attention2 * feats[2]
        x_Block2_32x32 = self.downsample32x32(x_Block2_SA)

        attention3 = self.sa3(feats[3])
        x_Block3_SA = attention3 * feats[3]
        x_Block3_32x32 = self.downsample32x32(x_Block3_SA)

        x_concat = torch.cat((x_Block1_32x32, x_Block2_32x32, x_Block3_32x32), dim=1)
        map_x = self.lastconv1(x_concat)

        map_x = map_x.squeeze(1)
        x = self.conv_head(feats[-1])
        x = self.bn2(x)
        x = self.act2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.classifier(x)
        return x, map_x


def build_feat_model(**kwargs):
    return FeatModel(**kwargs)
