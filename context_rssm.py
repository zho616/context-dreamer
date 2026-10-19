"""Context-conditioned block-GRU, with recurrent concatenation by default.

Recurrent input conditioning follows cRSSM and DALI's ``RSSM.img_step``:
https://github.com/frankroeder/DALI/blob/main/dreamerv3_compat/dreamerv3/nets.py
Here context joins the stochastic input projection of the existing block-GRU.
This transfers the recurrent conditioning route, not either complete method.
"""

import torch
from torch import nn

from rssm import RSSM
from tools import weight_init_


class ModulatedDeter(nn.Module):
    """Modulate each pre-GRU hidden block; zero initialization equals base Deter.

    Reuse RMSNorm and original GRU gates. The extra gain starts at one, unlike
    the paper's Transformer residual gate, since these are feed-forward blocks.
    Context is target-time c_t, matching the concatenation path's convention.
    """
    def __init__(self, base, context_dim=8):
        super().__init__()
        self.base = base
        width = base._dyn_in0[0].in_features
        self.modulations = nn.ModuleList([nn.Linear(context_dim, 3 * width) for _ in range(base.dynlayers)])
        for layer in self.modulations:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, stoch, deter, action, context):
        base = self.base
        action = action / torch.clip(torch.abs(action), min=1.0).detach()
        x = torch.cat((base._dyn_in0(deter), base._dyn_in1(stoch.reshape(action.shape[0], -1)),
                       base._dyn_in2(action)), -1)
        x = x.unsqueeze(-2).expand(-1, base.blocks, -1)
        x = base.group2flat(torch.cat((base.flat2group(deter), x), -1))
        for i, modulation in enumerate(self.modulations):
            x = base._dyn_hid.get_submodule(f"dyn_hid_{i}")(x)
            x = base._dyn_hid.get_submodule(f"norm_{i}")(x)
            scale, shift, gain = modulation(context.detach().to(x.dtype)).chunk(3, -1)
            x = base._dyn_hid.get_submodule(f"act_{i}")((1 + scale) * x + shift) * (1 + gain)
        gates = base.flat2group(base._dyn_gru(x)).chunk(3, -1)
        reset, cand, update = (base.group2flat(x) for x in gates)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1)
        return update * cand + (1 - update) * deter


class ContextRSSM(RSSM):
    def __init__(self, config, embed_size, act_dim, injection="recurrent_concat"):
        super().__init__(config, embed_size, act_dim)
        self.injection = injection
        if injection == "recurrent_concat":
            first = self._deter_net._dyn_in1[0]
            layer = nn.Linear(first.in_features + 8, first.out_features)
            layer.apply(weight_init_)
            self._deter_net._dyn_in1[0] = layer
        elif injection == "adarms_deter":
            self._deter_net = ModulatedDeter(self._deter_net)
        elif injection == "prior_posterior":
            for net in (self._obs_net, self._img_net):
                name, first = next(iter(net.named_children()))
                layer = nn.Linear(first.in_features + 8, first.out_features)
                layer.apply(weight_init_)
                net._modules[name] = layer
        else:
            raise ValueError(f"Unknown context injection: {injection}")

    def _transition(self, stoch, deter, action, context):
        if self.injection == "recurrent_concat":
            flat = stoch.reshape(action.shape[0], -1)
            inputs = torch.cat((flat, context.detach().to(flat.dtype)), -1)
            return self._deter_net(inputs, deter, action)
        if self.injection == "adarms_deter":
            return self._deter_net(stoch, deter, action, context)
        return self._deter_net(stoch, deter, action)

    def observe(self, embed, action, initial, reset, context):
        stoch, deter = initial
        stochs, deters, logits = [], [], []
        for i in range(action.shape[1]):
            stoch, deter, logit = self.obs_step(
                stoch, deter, action[:, i], embed[:, i], reset[:, i], context[:, i])
            stochs.append(stoch)
            deters.append(deter)
            logits.append(logit)
        return torch.stack(stochs, 1), torch.stack(deters, 1), torch.stack(logits, 1)

    def obs_step(self, stoch, deter, prev_action, embed, reset, context):
        reset = reset.bool().reshape(-1)
        stoch = torch.where(reset[:, None, None], 0., stoch)
        deter = torch.where(reset[:, None], 0., deter)
        prev_action = torch.where(reset[:, None], 0., prev_action)
        context = torch.where(reset[:, None], 0., context.detach())
        deter = self._transition(stoch, deter, prev_action, context)
        inputs = (deter, embed)
        if self.injection == "prior_posterior":
            inputs += (context.to(embed.dtype),)
        logit = self._obs_net(torch.cat(inputs, -1))
        return self.get_dist(logit).rsample(), deter, logit

    def prior(self, deter, context):
        # Recurrent modes already condition the shared deterministic state.
        inputs = deter
        if self.injection == "prior_posterior":
            inputs = torch.cat((deter, context.detach().to(deter.dtype)), -1)
        logit = self._img_net(inputs)
        return self.get_dist(logit).rsample(), logit

    def img_step(self, stoch, deter, prev_action, context):
        deter = self._transition(stoch, deter, prev_action, context)
        stoch, _ = self.prior(deter, context)
        return stoch, deter
