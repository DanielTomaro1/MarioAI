# model.py
import torch
import torch.nn as nn

class MarioNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # Handle both standard and frame-stacked inputs
        if len(input_dim) == 3:
            c, h, w = input_dim
        else:
            n_frames, c, h, w = input_dim
            c = n_frames  # Use number of frames as channels

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, input):
        # Handle frame-stacked input
        if len(input.shape) == 5:  # [batch, frames, channel, height, width]
            input = input.squeeze(2)  # Remove the channel dimension
        return self.online(input)