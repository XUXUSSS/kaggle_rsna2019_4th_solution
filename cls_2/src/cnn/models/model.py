import torch
import torch.nn as nn

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ELU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SimpleNet(nn.Module):
    def __init__(self,in_channel=1296, num_classes=6):
        super(SimpleNet, self).__init__()

        p = 0.3
        m = 4
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=100*m, kernel_size=3, stride=1, padding=1)
        self.elu1 = nn.ELU()

        self.conv2 = nn.Conv1d(in_channels=100*m, out_channels=100*m, kernel_size=3, stride=1, padding=1)
        self.elu2 = nn.ELU()

        self.conv3 = nn.Conv1d(in_channels=100*m, out_channels=200*m, kernel_size=3, stride=1, padding=1)
        self.elu3 = nn.ELU()

        self.conv4 = nn.Conv1d(in_channels=200*m, out_channels=200*m, kernel_size=3, stride=1, padding=1)
        self.se4 = SELayer(200*m)
        self.elu4 = nn.ELU()
        self.dropout = nn.Dropout(p)

        self.conv5 = nn.Conv1d(in_channels=200*m, out_channels=6, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        output = self.conv1(input)
        #print('input size: ' + str(input.size()))
        output = self.elu1(output)
        #print(output.size())

        output = self.conv2(output)
        output = self.elu2(output)
        #print(output.size())

        output = self.conv3(output)
        output = self.elu3(output)
        #print(output.size())

        output = self.conv4(output)
        output = self.se4(output)
        output = self.elu4(output)
        #print(output.size())
        output = self.dropout(output)

        output = self.conv5(output)

        #print(output.size())
        #output = output.view(-1,6,)
        #print(output.size())
        return output
