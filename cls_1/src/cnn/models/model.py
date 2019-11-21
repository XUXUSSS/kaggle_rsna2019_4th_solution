import torch
import torch.nn as nn



class SELayer1d(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer1d, self).__init__()
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


class SimpleNet1d(nn.Module):
    def __init__(self,in_channel=2064, num_classes=6):
        super(SimpleNet1d, self).__init__()

        p = 0.3
        m = 4
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=100*m, kernel_size=3, stride=1, padding=1)
        self.elu1 = nn.ELU()

        self.conv2 = nn.Conv1d(in_channels=100*m, out_channels=100*m, kernel_size=3, stride=1, padding=1)
        self.elu2 = nn.ELU()

        self.conv3 = nn.Conv1d(in_channels=100*m, out_channels=200*m, kernel_size=3, stride=1, padding=1)
        self.elu3 = nn.ELU()

        self.conv4 = nn.Conv1d(in_channels=200*m, out_channels=200*m, kernel_size=3, stride=1, padding=1)
        self.se4 = SELayer1d(200*m)
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



class SELayer3d(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer3d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ELU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class SimpleNet3d(nn.Module):
    def __init__(self,in_channel3d = 6, num_classes=6):
        super(SimpleNet3d, self).__init__()

        m3d = 1

        self.conv3d1 = nn.Conv3d(in_channels=in_channel3d, out_channels=100*m3d, kernel_size= (3,1,1), stride=1, padding=(1,0,0))
        self.relu3d1  = nn.ReLU()

        self.conv3d2 = nn.Conv3d(in_channels=100*m3d, out_channels=200*m3d, kernel_size= (1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.relu3d2 = nn.ReLU()


        self.conv3d3 = nn.Conv3d(in_channels=200*m3d, out_channels=200*m3d, kernel_size= (3,3,3), stride=(1,2,2), padding=(1,1,1))
        self.se3d3 = SELayer3d(200*m3d)
        self.relu3d3 = nn.ReLU()

        self.pooling2d = nn.AdaptiveAvgPool3d((None, 1, 1))

        # last classifier
        self.dropout = nn.Dropout(0.3)
        self.lastconv = nn.Conv1d(in_channels=200*m3d, out_channels=6, kernel_size=3, stride=1, padding=1)

    def forward(self, input3d):

        # branch 3D
        output3d = self.conv3d1(input3d)
        output3d = self.relu3d1(output3d)

        output3d = self.conv3d2(output3d)
        output3d = self.relu3d2(output3d)

        output3d = self.conv3d3(output3d)
        output3d = self.se3d3(output3d)
        output3d = self.relu3d3(output3d)

        output3d = self.pooling2d(output3d)


        output3d = torch.squeeze(output3d)


        output = self.dropout(output3d)

        output = self.lastconv(output)

        return output


class SimpleNet(nn.Module):
    def __init__(self,in_channel3d = 6, num_classes=6):
        super(SimpleNet, self).__init__()
        self.net3d = SimpleNet3d()
        self.net1d = SimpleNet1d()

    def forward(self, input1d, input3d):
        b, c, k, _, _ = input3d.size()


        output1d = self.net1d(input1d)
        input3d = input3d + (output1d.view(b, c, k, 1, 1)).expand_as(input3d)
        output = self.net3d(input3d) + output1d

        return output



if __name__ == "__main__":
    import torch
    import logging
    logging.getLogger().setLevel(logging.DEBUG)
    # ---------
    net = SimpleNet()
    print (net)
    data1 = torch.autograd.Variable(torch.randn(2, 2054, 32))
    data2 = torch.autograd.Variable(torch.randn(2,    6, 32, 14, 14))
    output = net(data1, data2)
    torch.save({'state_dict': net.state_dict()}, './tmp.pth')
    print (output.shape)

