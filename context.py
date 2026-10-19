"""Deterministic K5 ego context and auxiliary residual dynamics (PyTorch).

Port of Adaptive-Context-Dreamer 8e9c19a97ae8, rich dynamics path.
History tokens are [endpoint state, preceding executed canonical action2].
State: longitudinal acceleration, right-positive yaw rate, planar speed and
right-positive measured front-wheel angle. Legacy two-state histories remain supported.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

STATE_SCALE = (10., 1., 30., math.pi / 3)
STATE_DIM = len(STATE_SCALE)
ACTION_DIM = 2
TOKEN_DIM = STATE_DIM + ACTION_DIM
CONTEXT_DIM = 8
HISTORY_LENGTH = 5
FEATURE_SCALE = STATE_SCALE + (1., 1.)


def state_scale(state_dim):
    if state_dim not in (2, 4):
        raise ValueError("Context supports legacy state2 or speed/wheel-angle state4")
    return STATE_SCALE[:state_dim]


def predict_next_state(predictor, context, state, action):
    next_state = state.float() + predictor(context, state, action)
    if state.shape[-1] == 4:
        # Speed is a magnitude; MetaDrive's global wheel-angle bound is 60 deg.
        next_state = torch.cat((next_state[..., :2], next_state[..., 2:3].clamp_min(0),
                                next_state[..., 3:4].clamp(-math.pi / 3, math.pi / 3)), -1)
    return next_state


def canonical_action(action):
    """MetaDrive [left steering, throttle/brake] -> [throttle/brake, right steering]."""
    action = action.float().clamp(-1, 1)
    return torch.stack((action[..., 1], -action[..., 0]), -1)


class HistoryBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(64, eps=1e-5)
        self.qkv = nn.Linear(64, 192)
        self.out = nn.Linear(64, 64)
        self.norm2 = nn.LayerNorm(64, eps=1e-5)
        self.ff1 = nn.Linear(64, 128)
        self.ff2 = nn.Linear(128, 64)

    def forward(self, x, mask):
        batch, length, _ = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(batch, length, 3, 4, 16)
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        logits = (q @ k.transpose(-1, -2)) / 4.
        # Finite masking and final masking also handle entirely empty histories.
        weights = logits.masked_fill(~mask[:, None, None, :], -1e9).softmax(-1)
        attended = (weights @ v).transpose(1, 2).reshape(batch, length, 64)
        x = x + self.out(attended)
        x = x + self.ff2(F.gelu(self.ff1(self.norm2(x)), approximate="tanh"))
        return torch.where(mask[..., None], x, 0.)


class HistoryEncoder(nn.Module):
    def __init__(self, state_dim=STATE_DIM):
        super().__init__()
        self.token_dim = len(state_scale(state_dim)) + ACTION_DIM
        self.input = nn.Linear(self.token_dim, 64)
        self.position = nn.Parameter(torch.randn(HISTORY_LENGTH, 64) * .02)
        self.blocks = nn.ModuleList([HistoryBlock(), HistoryBlock()])
        self.mean = nn.Linear(64, CONTEXT_DIM)
        self.register_buffer("scale", torch.tensor(state_scale(state_dim) + (1., 1.)))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=1 / math.sqrt(module.in_features))
                nn.init.zeros_(module.bias)

    @torch.compiler.disable
    def forward(self, history, mask):
        # Keep this small FP32 attention branch eager. With torch 2.8 Inductor,
        # training-sized batches under an outer AMP region can lower attention
        # to FP16 despite the disabled autocast below, leaving FP32 projections.
        # Gradients still flow to E through the auxiliary loss across this break.
        if history.shape[-2:] != (HISTORY_LENGTH, self.token_dim) or mask.shape != history.shape[:-1]:
            raise ValueError(f"Expected history [...,5,{self.token_dim}] and mask [...,5]")
        shape = history.shape[:-2]
        with torch.autocast(device_type=history.device.type, enabled=False):
            mask = mask.bool().reshape(-1, HISTORY_LENGTH)
            h = history.float().reshape(-1, HISTORY_LENGTH, self.token_dim)
            h = torch.where(mask[..., None], h / self.scale, 0.)
            x = torch.where(mask[..., None], self.input(h) + self.position, 0.)
            for block in self.blocks:
                x = block(x, mask)
            pooled = x.sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            context = torch.where(mask.any(-1, keepdim=True), self.mean(pooled), 0.)
        return context.reshape(*shape, CONTEXT_DIM)


class ResidualDynamics(nn.Module):
    def __init__(self, state_dim=STATE_DIM):
        super().__init__()
        scales = state_scale(state_dim)
        self.hidden1 = nn.Linear(CONTEXT_DIM + state_dim + ACTION_DIM, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.delta = nn.Linear(128, state_dim)
        self.register_buffer("scale", torch.tensor(scales))

    def forward(self, context, state, action):
        with torch.autocast(device_type=state.device.type, enabled=False):
            x = torch.cat((context.float(), state.float() / self.scale, action.float()), -1)
            return self.delta(F.silu(self.hidden2(F.silu(self.hidden1(x))))) * self.scale


def auxiliary_loss(predictor, data, context):
    """Fixed anchor context, five steps, step-wise detached state, normalized L1.

    Unlike the legacy JAX replay, ctx_prev_action at target t+1 contains a_t.
    RSSM's shifted data['action'] is deliberately not used here.
    """
    states = data["ctx_ego_state"].float()
    actions = data["ctx_prev_action"].float()
    batch, length = states.shape[:2]
    if length < 2:
        raise ValueError("Context supervision needs at least two timesteps")
    flag = lambda key: data[key].bool().reshape(batch, length)
    event = flag("is_terminal") | flag("is_last") | flag("ctx_collision")
    alive = ~event & torch.isfinite(states).all(-1) & torch.isfinite(context).all(-1)
    predicted = torch.where(alive[..., None], states, 0.)
    ctx = torch.where(alive[..., None], context, 0.)
    scale = states.new_tensor(state_scale(states.shape[-1]))
    total = states.new_zeros(())
    count = torch.zeros((), device=states.device, dtype=torch.long)
    for horizon in range(1, min(5, length - 1) + 1):
        support = length - horizon
        target = states[:, horizon:]
        action = actions[:, horizon:]
        alive = alive[:, :support] & ~(event | flag("is_first"))[:, horizon:]
        alive = alive & torch.isfinite(target).all(-1) & torch.isfinite(action).all(-1)
        x = torch.where(alive[..., None], predicted[:, :support].detach(), 0.)
        a = torch.where(alive[..., None], action.clamp(-1, 1), 0.)
        predicted = predict_next_state(predictor, ctx[:, :support], x, a)
        target = torch.where(alive[..., None], target, 0.)
        error = ((predicted - target).abs() / scale).sum(-1)
        total = total + torch.where(alive, error, 0.).sum()
        count = count + alive.sum()
    return total / count.clamp_min(1), count


@torch.no_grad()
def refresh_context(encoder, predictor, state, action, context, history, mask, alive):
    """Predict endpoint before encoding target-time context for the RSSM."""
    action = canonical_action(action)
    next_state = predict_next_state(predictor, context, state, action)
    token = torch.cat((next_state, action), -1)
    history = torch.cat((history[..., 1:, :], token[..., None, :]), -2)
    mask = torch.cat((mask[..., 1:], alive[..., None]), -1) & alive[..., None]
    history = torch.where(mask[..., None], history, 0.)
    context = encoder(history, mask)
    return torch.where(alive[..., None], next_state, 0.), context, history, mask
