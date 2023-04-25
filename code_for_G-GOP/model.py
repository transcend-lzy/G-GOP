from utils import *


# VAE模型
class VAE_NEW(nn.Module):
    def __init__(self, latent_dim, dim=32*5):
        super(VAE_NEW, self).__init__()
        self.x = 0
        self.latent_dim = latent_dim
        self.dim = dim
        self.liner_dim = 100
        #网络结构
        self.bn = nn.BatchNorm2d(self.dim * 16)
        self.fc1 = nn.Linear(latent_dim, self.liner_dim) #100维全连接层
        self.fc2 = nn.Linear(self.liner_dim, self.liner_dim) #100维全连接层
        self.fc3 = nn.Linear(self.liner_dim, 4*4*self.dim*16)


        self.fc4 = nn.Sequential(
            nn.ConvTranspose2d(self.dim*16, self.dim*8, 5, 2, 2, 1, bias=False),  #padding
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.dim*8)
        )
        self.fc5 = nn.Sequential(
            nn.ConvTranspose2d(self.dim*8, self.dim*4, 5, 2, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.dim*4)
        )
        self.fc6 = nn.Sequential(
            nn.ConvTranspose2d(self.dim*4, self.dim*2, 5, 2, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.dim*2)
        )

        self.fc7 = nn.Sequential(
            nn.ConvTranspose2d(self.dim*2, self.dim, 5, 2, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.dim)
        )
        #输入是kernel_size=5，stride = 2，padding为1
        self.fc8 = nn.ConvTranspose2d (self.dim, 1, 5, 2, 2, 1, bias=False)

    def front(self, x):  #获取学习中间特征，编码过程
        h = F.relu(self.fc1(x))
        h1 = F.relu(self.fc2(h))
        h2 = F.relu(self.fc3(h1))
        h3 = h2.view(-1, self.dim * 16, 4, 4)  # 通道数是self.dim * 16
        res = self.bn(h3)
        return res

    def decode(self, x):  #解码生成图像
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        return torch.sigmoid(self.fc8(x)).view(-1,128,128)
 
    # 计算重构值和隐变量z的分布参数
    def forward(self, x):
        self.x = x
        z = self.front(x)
        return self.decode(z)