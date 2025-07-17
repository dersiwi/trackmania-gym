import torch
import torch.nn as nn



class VisionModelSix(nn.Module): #seal team 6 very cool
    def __init__(self, in_color_channels : int, out_dim : int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_color_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample by 2

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
          #  nn.AdaptiveAvgPool2d((4, 4)),  # output fixed spatial size: (128, 4, 4)
        )


        self.flatten = nn.Flatten()
        self.output_layer = nn.Sequential(
            nn.Linear(128 * 16 * 16, out_dim), #TODO this is hard coded, think of a way to change this 
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.output_layer(x)
        return x

class VisionModelEight(nn.Module):
    def __init__(self, in_color_channels : int, out_dim=256):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            )

        self.encoder = nn.Sequential(
            conv_block(in_color_channels, 32),
            conv_block(32, 32),
            nn.MaxPool2d(2),  # Down H,W -> H/2, W/2

            conv_block(32, 64),
            conv_block(64, 64),
            nn.MaxPool2d(2),  # H/4, W/4

            conv_block(64, 128),
            conv_block(128, 128),
            nn.Dropout(0.2)
        )


        with torch.no_grad():
            dummy_input = torch.zeros(1, in_color_channels, *(128, 128)) #TODO terrible to hardcode images size here.
            dummy_out = self.encoder(dummy_input)
            self.flatten_dim = dummy_out.view(1, -1).shape[1]

        self.feature_projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.feature_projector(x)
        return x 

class Linesight_Vision_Model(nn.Module):
    """
    This is the same vision model which was used in linesight_rl 
    """
    def __init__(self,in_color_channels=1,img_head_channels = [1, 16, 32, 64, 32]):
        super(Linesight_Vision_Model,self).__init__()
        img_head_channels[0] = in_color_channels
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=img_head_channels[0], out_channels=img_head_channels[1], kernel_size=(4, 4), stride=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=img_head_channels[1], out_channels=img_head_channels[2], kernel_size=(4, 4), stride=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=img_head_channels[2], out_channels=img_head_channels[3], kernel_size=(3, 3), stride=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=img_head_channels[3], out_channels=img_head_channels[4], kernel_size=(3, 3), stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Flatten(),
        )
    def forward(self,x):
        return self.model(x)
    