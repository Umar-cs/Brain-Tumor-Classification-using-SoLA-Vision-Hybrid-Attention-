import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from einops import rearrange


root_path = '/kaggle/input/brisc2025/brisc2025/classification_task/train'

image_paths = []
labels = []

for label in os.listdir(root_path):
    label_path = os.path.join(root_path, label)
    if os.path.isdir(label_path):
        for img_file in os.listdir(label_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(label_path, img_file))
                labels.append(label)

df = pd.DataFrame({'image_path': image_paths, 'label': labels})


train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df['label'], random_state=42
)

# Oversampling
max_samples = train_df['label'].value_counts().max()

balanced_train_df = train_df.groupby('label', group_keys=False).apply(
    lambda x: x.sample(n=max_samples, replace=True, random_state=42)
).reset_index(drop=True)


class BrainTumorDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.label_map = {
            'pituitary': 0,
            'no_tumor': 1,
            'meningioma': 2,
            'glioma': 3
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df['image_path'].iloc[idx]
        label = self.label_map[self.df['label'].iloc[idx]]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

train_dataset = BrainTumorDataset(balanced_train_df, transform_train)
val_dataset = BrainTumorDataset(val_df, transform_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        h = self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        k = k.softmax(dim=-1)

        context = torch.einsum('b h n d, b h n e -> b h d e', k, v)
        out = torch.einsum('b h d e, b h n d -> b h n e', context, q)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class SoLABlock(nn.Module):
    def __init__(self, dim, heads=4, use_softmax=False):
        super().__init__()
        self.use_softmax = use_softmax
        self.norm1 = nn.LayerNorm(dim)

        if use_softmax:
            self.attn = nn.MultiheadAttention(dim, heads)
        else:
            self.attn = LinearAttention(dim, heads)

        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        residual = x
        attn_input = self.norm1(x)

        if self.use_softmax:
            attn_input = attn_input.permute(1, 0, 2)
            out, _ = self.attn(attn_input, attn_input, attn_input)
            out = out.permute(1, 0, 2)
        else:
            out = self.attn(attn_input)

        x = residual + out
        x = x + self.ffn(self.norm2(x))

        return x



class SoLAVision(nn.Module):
    def __init__(self, num_classes=4, dim=256, depth=6, heads=4):
        super().__init__()

        self.patch_embed = nn.Conv2d(3, dim, kernel_size=16, stride=16)

        num_patches = (224 // 16) ** 2

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        pattern = ['L', 'L', 'L', 'S', 'L', 'L']

        self.blocks = nn.ModuleList([
            SoLABlock(dim, heads, use_softmax=(p == 'S')) for p in pattern
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        b = x.shape[0]

        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x[:, 0])
        return self.head(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SoLAVision().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20

for epoch in range(num_epochs):
    model.train()

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs} completed")


model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.numpy())
        y_pred.extend(predicted.cpu().numpy())

print(classification_report(y_true, y_pred))