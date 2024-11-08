"""
Author: Mélanie Gaillochet
Code for "Automating MedSAM by Learning Prompts with Weak Few-Shot Supervision", Gaillochet et al. (MICCAI-MedAGI, 2024)
"""
import torch.nn as nn
from torch.nn import functional as F

    
class SimplePromptModule(nn.Module):
    def __init__(self, **kwargs):
        super(SimplePromptModule, self).__init__()
        self.num_points = kwargs.get('num_points', 1)
        self.num_center_channels = kwargs.get('num_center_channels', 512)

        # Convolutional layer to get the mask embed (first a 1x1 convolution to reduce the number of channels)
        self.conv1 = nn.Conv2d(256, self.num_center_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(self.num_center_channels, 256, kernel_size=3, stride=1, padding=1)

        # Layers to get the points embed
        self.conv3 = nn.Conv2d(self.num_center_channels, self.num_center_channels//2, kernel_size=1, stride=1, padding=0) # Reduce the number of channels to  self.num_center_channels//2
        maxpool_kernel = 8
        self.max_pool = nn.MaxPool2d(maxpool_kernel, stride=maxpool_kernel) # Max Pooling to reduce feature map H and W to 1/maxpool_kernel

        # Fully connected layer to transform flattened [BS, self.num_center_channels//2x8x8] to [BS, 256*(num_points+1)]
        self.fc = nn.Linear(self.num_center_channels//2*64//maxpool_kernel*64//maxpool_kernel, 256*(self.num_points+1))

    def forward(self, x):
        # Apply convolutional layers
        out = F.relu(self.conv1(x)) # [BS, num_center_channels, 64, 64]
        mask_embed = F.relu(self.conv2(out)) # [BS, 256, 64, 64]

        # Max Pooling to reduce each feature map --> # [BS, num_center_channels//2, 8, 8]
        out = F.relu(self.conv3(out)) # [BS, num_center_channels//2, 64, 64]
        out = self.max_pool(out) # [BS, num_center_channels//2, 8, 8]
        
        # We apply fully connected
        flat_out = out.flatten(start_dim=1)
        _points_embed = self.fc(flat_out)

        # Reshape to the desired output shape [BS, 1, num_points + 1, 256]
        points_embed = _points_embed.view(-1, 1, self.num_points + 1, 256) # [BS, 1, num_points + 1, 256]

        return points_embed, mask_embed, None


promptmodule_zoo = {
            "simple_module": SimplePromptModule,
            }
