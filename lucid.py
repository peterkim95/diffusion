from torchvision.datasets import MNIST, CelebA
import torchvision.transforms as transforms

from unet import Unet, GaussianDiffusion, Trainer

IMG_SIZE = 32

model = Unet(
    dim = 16,
    dim_mults = (1, 2, 4, 8),
    channels = 1
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = IMG_SIZE,
    channels=1,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()

img_transform = transforms.Compose([transforms.Resize(IMG_SIZE),
                                    transforms.CenterCrop(IMG_SIZE),
                                    transforms.ToTensor()])
mnist_dataset = MNIST(root='./data/MNIST', train=True, transform=img_transform, download=True)


trainer = Trainer(
    diffusion,
    mnist_dataset,
    train_batch_size = 32,
    train_lr = 1e-4,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True                        # turn on mixed precision
)

trainer.train()
