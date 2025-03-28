import torch
import torch.nn as nn


# Self-Attention層 (変更なし)
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Attention weight
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, height, width = x.size()

        # (bs, c, h, w) -> (bs, c/8, h, w) -> (bs, c/8, h*w) -> (bs, h*w, c/8)
        proj_query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        # (bs, c, h, w) -> (bs, c/8, h, w) -> (bs, c/8, h*w)
        proj_key = self.key(x).view(batch_size, -1, height * width)

        # バッチごとに行列積を計算 -> (bs, h*w, h*w)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)  # B x (H*W) x (H*W)

        proj_value = self.value(x).view(batch_size, -1, height * width)  # B x C x (H*W)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1)).view(batch_size, C, height, width)
        return self.gamma * out + x, attention


# CNNモデルの定義 (pool1とpool3を除去)
class Net(nn.Module):
    def __init__(self, apply_attention=True):
        super(Net, self).__init__()
        self.apply_attention = apply_attention

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # self.relu3 = nn.ReLU()

        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

        self.attention_mlp = nn.Sequential(
            nn.Linear(256, 64 * 8 * 8, bias=True),
            nn.BatchNorm1d(64 * 8 * 8, affine=True, momentum=0.9),
            nn.ReLU(),
            nn.Linear(64 * 8 * 8, 64 * 8 * 8)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_feat=False, return_attention=False):
        # (bs, 3, 32, 32) -> (bs, 32, 16, 16)
        conv1_out = self.relu1(self.conv1(x))
        pool1_out = self.pool1(conv1_out)

        # (bs, 32, 16, 16) -> (bs, 64, 8, 8)
        conv2_out = self.relu2(self.conv2(pool1_out))
        pool2_out = self.pool2(conv2_out)

        bs, ch, h, w = pool2_out.size()  # 例: bs, 64, 8, 8

        if self.apply_attention:
            ### attention map の計算
            # (bs, c, h, w) -> (bs, c * h * w)
            flattened_pool2 = pool2_out.view(bs, -1)
            # attention mapを生成するために、全結合層の出力を計算
            temp_fc1_out = self.relu4(self.fc1(flattened_pool2))
            # MLPでattention mapを計算
            attention_map = self.attention_mlp(temp_fc1_out)
            attention_map = self.sigmoid(attention_map)
            # reshapeして (bs, c, h, w) に戻す
            attention_map = attention_map.view(bs, ch, h, w)

            # attention mapを用いてpool2_outに乗算
            attended_pool2 = pool2_out * attention_map

            flattened = attended_pool2.view(bs, -1)
            fc1_out = self.relu4(self.fc1(flattened))
            logits = self.fc2(fc1_out)

        else:
            flattened = pool2_out.view(bs, -1)
            fc1_out = self.relu4(self.fc1(flattened))
            logits = self.fc2(fc1_out)

        if return_feat:
            return logits, fc1_out
        
        elif return_attention:
            return logits, attention_map
        
        else:
            return logits
