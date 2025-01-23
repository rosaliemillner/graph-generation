import torch
import torch.nn as nn
from blocks import get_time_embedding
from blocks import DownBlock, MidBlock, UpBlockUnet

DOWN_CHANNELS = [64, 32]
MID_CHANNELS = [32, 64]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DenoisingUnet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.down_channels = DOWN_CHANNELS
        self.mid_channels = MID_CHANNELS
        self.t_emb_dim = hidden_dim
        self.down_sample = [2]
        self.num_down_layers = 2
        self.num_mid_layers = 2
        self.num_up_layers = 2
        self.attns = [2]
        self.norm_channels = 2
        self.num_heads = 2
        self.fc_out_dim = hidden_dim

        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-2]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1

        # Initial projection from sinusoidal time embedding
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )

        self.up_sample = list(reversed(self.down_sample))
        self.fc_in = nn.Linear(input_dim, self.down_channels[0])

        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.downs.append(DownBlock(self.down_channels[i], self.down_channels[i + 1], self.t_emb_dim, 2))

        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i + 1], self.t_emb_dim,
                                      num_heads=self.num_heads,
                                      num_layers=self.num_mid_layers,
                                      ))

        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels) - 1)):
            self.ups.append(UpBlockUnet(self.down_channels[i] * 2, self.down_channels[i - 1] if i != 0 else self.fc_out_dim,
                                        self.t_emb_dim, up_sample=self.down_sample[i],
                                        num_heads=self.num_heads,
                                        num_layers=self.num_up_layers,
                                        ))

        self.norm_out = nn.GroupNorm(self.norm_channels, self.fc_out_dim)
        self.fc_out = nn.Linear(self.fc_out_dim, input_dim)

    def forward(self, x, t, cond):
        x = x.to(device)
        out = self.fc_in(x)

        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)

        down_outs = []

        for idx, down in enumerate(self.downs):
            down_outs.append(out)
            out = down(out, t_emb)

        for mid in self.mids:
            out = mid(out, t_emb)

        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb)
        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.fc_out(out)
        return out
