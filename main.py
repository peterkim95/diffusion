import torch
from torch import nn
from torchvision.datasets import MNIST, CelebA
import torchvision.transforms as transforms
from tqdm import trange

from unet import Unet, Trainer


def unsqueeze_like(t, like_t):
    return t.reshape(*t.shape, *(1,) * (len(like_t.shape) - len(t.shape)))


class Diffusion(nn.Module):
    def __init__(self, denoiser, num_timesteps, img_size, channels):
        super().__init__()
        self.denoiser = denoiser
        self.num_timesteps = num_timesteps
        self.img_size = img_size
        self.channels = channels

        self.loss = nn.MSELoss()

        beta_1 = 10e-4
        beta_T = 0.02
        betas = torch.linspace(beta_1, beta_T, steps=num_timesteps)
        alpha = 1 - betas

        self.sigma = torch.sqrt(betas)
        self.one_over_sqrt_alpha = 1 / torch.sqrt(alpha)
        self.alpha_cumprod = torch.cumprod(alpha, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_root_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod)

        self.beta_over_sqrt_root_one_minus_alpha_cumprod = betas / self.sqrt_root_one_minus_alpha_cumprod

        # self.prev_alpha_cumprod = F.pad(self.alpha_cumprod[:-1], pad=(1,0), value=1)
        # self.posterior_variance = ((1-self.prev_alpha_cumprod) / (1-alpha)) * betas

    def q_sample(self, x_0, t, noise):  # q(x_t | x_0)
        b, *_ = x_0.shape
        x_t = unsqueeze_like(self.sqrt_alpha_cumprod[t], x_0) * x_0 \
              + unsqueeze_like(self.sqrt_root_one_minus_alpha_cumprod[t], x_0) * noise
        return x_t

    def get_simple_loss(self, x_0):
        b, c, h, w = x_0.shape
        t = torch.randint(self.num_timesteps, (b,))

        target_noise = torch.randn_like(x_0)  # N(0,I)
        x_t = self.q_sample(x_0, t, target_noise)
        predicted_noise = self.denoiser(x_t, t)

        return self.loss(predicted_noise, target_noise)

    def p_sample(self, x_t, t):  # p(x_(t-1) | x_t)
        predicted_noise = self.denoiser(x_t, t)  # (B, C, W, H)
        sample_noise = torch.randn_like(x_t)
        # TODO: uglier than i expected
        x_t_minus_1 = unsqueeze_like(self.one_over_sqrt_alpha[t], predicted_noise) \
                      * (x_t - unsqueeze_like(self.beta_over_sqrt_root_one_minus_alpha_cumprod[t],
                                              predicted_noise) * predicted_noise) \
                      + unsqueeze_like(self.sigma[t], sample_noise) * sample_noise
        return x_t_minus_1

    def sample(self, sample_batch_size=8):
        img_shape = (sample_batch_size, self.channels, self.img_size, self.img_size)
        x_t = torch.randn(img_shape)
        for t in trange(self.num_timesteps - 1, -1, -1):
            t = torch.ones(sample_batch_size, dtype=torch.long) * t
            x_t = self.p_sample(x_t, t)
        return x_t


# TODO: How do img dims all relate to one another, between Unet and Diffusion
IMG_SIZE = 128
CHANNELS = 1

denoiser = Unet(dim=64, dim_mults=(1, 2, 4, 8), channels=CHANNELS)

diffuser = Diffusion(denoiser=denoiser, num_timesteps=1000, img_size=IMG_SIZE, channels=CHANNELS)

img_transform = transforms.Compose([transforms.Resize(IMG_SIZE),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.CenterCrop(IMG_SIZE),
                                    transforms.ToTensor()])

mnist_dataset = MNIST(root='./data/MNIST', train=True, transform=img_transform, download=True)
# celeba_dataset = CelebA(root='./data/CelebA', split='all', transform=img_transform, download=True)

trainer = Trainer(
    diffusion_model=diffuser,
    dataset=mnist_dataset,
    train_batch_size=32,
    train_lr=1e-4,
    train_num_steps=700000,  # total training steps
    gradient_accumulate_every=2,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    amp=False  # turn on mixed precision
)

trainer.train()
