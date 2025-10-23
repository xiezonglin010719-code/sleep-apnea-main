import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import numpy as np


class Generator(nn.Module):
    """生成器：固定输出224x224尺寸的梅尔频谱图像"""

    def __init__(self, input_dim=100, output_channels=1):  # 移除img_size参数
        super(Generator, self).__init__()
        self.output_channels = output_channels

        self.model = nn.Sequential(
            # 输入：(batch, 100, 1, 1)
            nn.ConvTranspose2d(input_dim, 512, 7, 1, 0, bias=False),  # (batch, 512, 7, 7)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # (batch, 256, 14, 14)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # (batch, 128, 28, 28)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # (batch, 64, 56, 56)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),  # (batch, 32, 112, 112)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, output_channels, 4, 2, 1, bias=False),  # (batch, 1, 224, 224)
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        img = self.model(z)
        return img


class Discriminator(nn.Module):
    """判别器：适配224x224输入尺寸"""

    def __init__(self, input_channels=1):  # 移除img_size参数
        super(Discriminator, self).__init__()
        self.input_channels = input_channels

        self.model = nn.Sequential(
            # 输入：(batch, 1, 224, 224)
            nn.Conv2d(input_channels, 16, 4, 2, 1, bias=False),  # (batch, 16, 112, 112)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, 32, 4, 2, 1, bias=False),  # (batch, 32, 56, 56)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1, bias=False),  # (batch, 64, 28, 28)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # (batch, 128, 14, 14)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # (batch, 256, 7, 7)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, 7, 1, 0, bias=False)  # (batch, 1, 1, 1)
        )

    def forward(self, img):
        validity = self.model(img)
        return validity.view(-1, 1)


class WGANGP(nn.Module):
    """WGAN-GP整体模块：移除img_size参数"""

    def __init__(self, input_dim=100, img_channels=1, device='cpu'):  # 移除img_size参数
        super(WGANGP, self).__init__()
        self.input_dim = input_dim
        self.device = device

        # 初始化生成器和判别器（不再传递img_size）
        self.generator = Generator(input_dim=input_dim, output_channels=img_channels).to(device)
        self.discriminator = Discriminator(input_channels=img_channels).to(device)

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
        self.lambda_gp = 10

    def compute_gradient_penalty(self, real_samples, fake_samples):
        alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        fake = Variable(torch.ones_like(d_interpolates), requires_grad=False).to(self.device)
        gradients = torch_grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train_step(self, real_imgs, n_critic=5):
        batch_size = real_imgs.size(0)

        # 训练判别器
        self.discriminator.train()
        self.generator.eval()
        for _ in range(n_critic):
            self.optimizer_D.zero_grad()
            real_validity = self.discriminator(real_imgs)
            z = torch.randn(batch_size, self.input_dim).to(self.device)
            fake_imgs = self.generator(z)
            fake_validity = self.discriminator(fake_imgs)
            gradient_penalty = self.compute_gradient_penalty(real_imgs.data, fake_imgs.data)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_gp * gradient_penalty
            d_loss.backward()
            self.optimizer_D.step()

        # 训练生成器
        self.discriminator.eval()
        self.generator.train()
        self.optimizer_G.zero_grad()
        z = torch.randn(batch_size, self.input_dim).to(self.device)
        fake_imgs = self.generator(z)
        fake_validity = self.discriminator(fake_imgs)
        g_loss = -torch.mean(fake_validity)
        g_loss.backward()
        self.optimizer_G.step()

        return d_loss.item(), g_loss.item()

    def generate_data(self, n_samples, label=None):
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.input_dim).to(self.device)
            fake_imgs = self.generator(z)
            fake_imgs = (fake_imgs + 1) / 2 * 255.0  # 从[-1,1]转换为[0,255]
            fake_imgs = fake_imgs.clamp(0, 255).cpu().numpy().astype(np.uint8)
            fake_imgs = fake_imgs.squeeze(1)  # 移除通道维度
        if label is None:
            labels = np.random.randint(0, 2, size=n_samples)  # OSA二分类标签
        else:
            labels = np.full(n_samples, label)
        return fake_imgs, labels