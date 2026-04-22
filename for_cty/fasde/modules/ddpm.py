import sys
import torch
import torch.nn as nn
import numpy as np


class DDPM(nn.Module):
    def __init__(self, T=1000, beta_min=1e-6, beta_max=1e-2):
        super().__init__()
        self.T = T
        self.beta_min = beta_min
        self.beta_max = beta_max

        self.set_noise_schedule()


    def set_noise_schedule(self,):
        """
        Sets sampling noise schedule. Authors in the paper showed
        that WaveGrad supports variable noise schedules during inference.
        Thanks to the continuous noise level conditioning.
        :param init (callable function, optional): function which initializes betas
        :param init_kwargs (dict, optional): dict of arguments to be pushed to `init` function.
            Should always contain the key `steps` corresponding to the number of iterations to be done by the model.
            This is done so because `torch.linspace` has this argument named as `steps`.
        """
        # assert 'steps' in list(init_kwargs.keys()), \
        #     '`init_kwargs` should always contain the key `steps` corresponding to the number of iterations to be done by the model.'
        # n_iter = init_kwargs['steps']

        # betas = init(**init_kwargs)
        betas = torch.linspace(start=self.beta_min, end=self.beta_max, steps=self.T)
        alphas = 1 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        # alphas_cumprod_prev = torch.cat([torch.FloatTensor([1]), alphas_cumprod[:-1]])
        # alphas_cumprod_prev_with_last = torch.cat([torch.FloatTensor([1]), alphas_cumprod])
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        # self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Calculations for posterior q(y_n|y_0)
        sqrt_alphas_cumprod = alphas_cumprod.sqrt()
        sqrt_1m_alphas_cumprod = (1. - alphas_cumprod).sqrt()
        # For WaveGrad special continuous noise level conditioning
        # self.sqrt_alphas_cumprod_prev = alphas_cumprod_prev_with_last.sqrt().numpy()
        # sqrt_recip_alphas_cumprod = (1 / alphas_cumprod).sqrt()
        # sqrt_alphas_cumprod_m1 = (1 - alphas_cumprod).sqrt() * sqrt_recip_alphas_cumprod
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_1m_alphas_cumprod', sqrt_1m_alphas_cumprod)
        # self.register_buffer('sqrt_recip_alphas_cumprod', sqrt_recip_alphas_cumprod)
        # self.register_buffer('sqrt_alphas_cumprod_m1', sqrt_alphas_cumprod_m1)

        # # Calculations for posterior q(y_{t-1} | y_t, y_0)
        # posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        # posterior_variance = torch.stack([posterior_variance, torch.FloatTensor([1e-20] * self.T)])
        # posterior_log_variance_clipped = posterior_variance.max(dim=0).values.log()
        # # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        # posterior_mean_coef1 = betas * alphas_cumprod_prev.sqrt() / (1 - alphas_cumprod)
        # posterior_mean_coef2 = (1 - alphas_cumprod_prev) * alphas.sqrt() / (1 - alphas_cumprod)
        # self.register_buffer('posterior_log_variance_clipped', posterior_log_variance_clipped)
        # self.register_buffer('posterior_mean_coef1', posterior_mean_coef1)
        # self.register_buffer('posterior_mean_coef2', posterior_mean_coef2)

    def sample_t(self, batch_size):
        t = np.random.choice(self.T, (batch_size,))
        return torch.LongTensor(t)

    def q_sample(self, y_0, t, eps = None):
        """
        Efficiently computes diffusion version y_t from y_0 using a closed form expression:
            y_t = sqrt(alpha_cumprod)_t * y_0 + sqrt(1 - alpha_cumprod_t) * eps,
            where eps is sampled from a standard Gaussian.
        """
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_1m_alpha_cumprod = self.sqrt_1m_alphas_cumprod[t]

        if eps is None:
            eps = torch.randn_like(y_0)
        y_t = sqrt_alpha_cumprod_t[..., None, None] * y_0 + sqrt_1m_alpha_cumprod[..., None, None] * eps
        return y_t

        # continuous_sqrt_alpha_cumprod \
        #     = self.sample_continuous_noise_level(batch_size, device=y_0.device) \
        #         if isinstance(eps, type(None)) else continuous_sqrt_alpha_cumprod
        # if isinstance(eps, type(None)):
        #     eps = torch.randn_like(y_0)
        # # Closed form signal diffusion
        # outputs = continuous_sqrt_alpha_cumprod * y_0 + (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * eps
        # return outputs

    def sample(self, score_net, data, xt):
        # xt should be initialized as noise
        ts = torch.arange(self.T-1, -1, -1).to(xt.device)
        batch_size = xt.shape[0]
        x0 = None
        gvp_encoder_output = None
        x0_trajectory, logits_trajectory = [], []
        for step, t in enumerate(ts):
            if step % 10 == 0:
                print('.', end='')
                sys.stdout.flush()
            
            t = t.view(batch_size, )
            xt = xt if step == 0 else self.q_sample(x0, t)
            score_net_output = score_net(
                coords = data['coords'],
                padding_mask = data['padding_mask'],
                confidence = data['confidence'],
                positions = data['positions'],
                t = t,
                xt = xt,
                gvp_encoder_output = gvp_encoder_output,
            )

            x0 = score_net_output['x0'].transpose(0, 1)
            logits = score_net_output['logits'].transpose(0, 1)
            gvp_encoder_output = (score_net_output['gvp_output'], score_net_output['encoder_embedding'][0])

            x0_trajectory.append(x0)
            logits_trajectory.append(logits)
        print('')

        x0_trajectory = torch.cat(x0_trajectory, dim=0)
        logits_trajectory = torch.cat(logits_trajectory, dim=0)

        return x0_trajectory, logits_trajectory


        