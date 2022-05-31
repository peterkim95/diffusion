import torch
from torch import nn
from torchvision.datasets import MNIST, CelebA
import torchvision.transforms as transforms
from tqdm import trange

from unet import Unet, Trainer


def unsqueeze_like(t, like_t):
    return t.reshape(*t.shape, *(1,) * (len(like_t.shape) - len(t.shape)))


def normalize_to_zero_and_one(t):
    return (t+1)/2


def normalize_to_neg_one_and_one(t):
    return 2*t - 1


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

        self.register_buffer('sigma', torch.sqrt(betas))
        self.register_buffer('one_over_sqrt_alpha', 1 / torch.sqrt(alpha))
        self.register_buffer('alpha_cumprod', torch.cumprod(alpha, dim=0))
        self.register_buffer('sqrt_alpha_cumprod', torch.sqrt(self.alpha_cumprod))
        self.register_buffer('sqrt_root_one_minus_alpha_cumprod', torch.sqrt(1 - self.alpha_cumprod))
        self.register_buffer('beta_over_sqrt_root_one_minus_alpha_cumprod', betas / self.sqrt_root_one_minus_alpha_cumprod)
        '''
        self.sigma = torch.sqrt(betas)
        self.one_over_sqrt_alpha = 1 / torch.sqrt(alpha)
        self.alpha_cumprod = torch.cumprod(alpha, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_root_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod)

        self.beta_over_sqrt_root_one_minus_alpha_cumprod = betas / self.sqrt_root_one_minus_alpha_cumprod
        '''

    def q_sample(self, x_0, t, noise):  # q(x_t | x_0)
        b, *_ = x_0.shape
        # print(x_0.device, self.sqrt_alpha_cumprod.device, self.sqrt_alpha_cumprod[t].device, noise.device)
        x_t = unsqueeze_like(self.sqrt_alpha_cumprod[t], x_0) * x_0 \
              + unsqueeze_like(self.sqrt_root_one_minus_alpha_cumprod[t], x_0) * noise
        return x_t

    def get_simple_loss(self, x_0):
        b, c, h, w = x_0.shape
        t = torch.randint(self.num_timesteps, (b,), device=x_0.device).long()

        target_noise = torch.randn_like(x_0)  # N(0,I)
        x_t = self.q_sample(normalize_to_neg_one_and_one(x_0), t, target_noise)
        predicted_noise = self.denoiser(x_t, t)

        return self.loss(predicted_noise, target_noise)

    @torch.no_grad()
    def p_sample(self, x_t, t):  # p(x_(t-1) | x_t)
        predicted_noise = self.denoiser(x_t, t)  # (B, C, W, H)
        sample_noise = torch.randn_like(x_t)
        # TODO: uglier than i expected
        x_t_minus_1 = unsqueeze_like(self.one_over_sqrt_alpha[t], predicted_noise) \
                      * (x_t - unsqueeze_like(self.beta_over_sqrt_root_one_minus_alpha_cumprod[t],
                                              predicted_noise) * predicted_noise) \
                      + unsqueeze_like(self.sigma[t], sample_noise) * sample_noise
        return x_t_minus_1
    
    @torch.no_grad()
    def sample(self, sample_batch_size=16):
        img_shape = (sample_batch_size, self.channels, self.img_size, self.img_size)
        device = self.sigma.device
        x_t = torch.randn(img_shape, device=device)
        for t in trange(self.num_timesteps - 1, -1, -1):
            t = torch.ones(sample_batch_size, dtype=torch.long, device=device) * t
            x_t = self.p_sample(x_t, t)
        return normalize_to_zero_and_one(x_t)


# TODO: How do img dims all relate to one another, between Unet and Diffusion
# TODO: normalize and unnormalize images
# TODO: train it on gpu
IMG_SIZE = 32
CHANNELS = 1

denoiser = Unet(dim=16, dim_mults=(1, 2, 4, 8), channels=CHANNELS).cuda()

diffuser = Diffusion(denoiser=denoiser, num_timesteps=1000, img_size=IMG_SIZE, channels=CHANNELS).cuda()

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
    amp=True,  # turn on mixed precision
    save_and_sample_every=1000
)

trainer.train()
