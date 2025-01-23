import torch
import torch.nn as nn


def get_time_embedding(time_steps, temb_dim):
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"

    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2))
    )

    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.t_emb_dim = t_emb_dim
        self.mlp_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(in_channels if i == 0 else out_channels),
                    nn.ReLU(),
                    nn.Linear(in_channels if i == 0 else out_channels, out_channels),
                )
                for i in range(n_layers)
            ]
        )
        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList([
                nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(self.t_emb_dim, out_channels)
                )
                for _ in range(n_layers)
            ])
        self.mlp_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(out_channels),
                    nn.ReLU(),
                    nn.Linear(out_channels, out_channels),
                )
                for _ in range(n_layers)
            ]
        )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Linear(in_channels if i == 0 else out_channels, out_channels)
                for i in range(n_layers)
            ]
        )
        self.down_sample_conv = nn.Identity()

    def forward(self, x, t_emb=None):
        out = x
        for i in range(self.n_layers):
            # Graph convolution block
            resnet_input = out
            out = self.mlp_first[i](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i](t_emb)
            out = self.mlp_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)

        # Downsample
        out = self.down_sample_conv(out)
        return out


class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads, num_layers, cross_attn=None, d_cond=None):
        super().__init__()
        self.num_layers = num_layers
        self.t_emb_dim = t_emb_dim
        self.context_dim = d_cond
        self.cross_attn = cross_attn
        self.mlp_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(in_channels if i == 0 else out_channels),
                    nn.ReLU(),
                    nn.Linear(in_channels if i == 0 else out_channels, out_channels),
                )
                for i in range(num_layers + 1)
            ]
        )

        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList([
                nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(t_emb_dim, out_channels)
                )
                for _ in range(num_layers + 1)
            ])
        self.mlp_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(out_channels),
                    nn.ReLU(),
                    nn.Linear(out_channels, out_channels),
                )
                for _ in range(num_layers + 1)
            ]
        )

        self.attention_norms = nn.ModuleList(
            [nn.LayerNorm(out_channels)
             for _ in range(num_layers)]
        )

        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
             for _ in range(num_layers)]
        )
        if self.cross_attn:
            assert d_cond is not None
            self.cross_attention_norms = nn.ModuleList(
                [nn.LayerNorm(out_channels)
                 for _ in range(num_layers)]
            )
            self.cross_attentions = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                 for _ in range(num_layers)]
            )
            self.context_proj = nn.ModuleList(
                [nn.Linear(d_cond, out_channels)
                 for _ in range(num_layers)]
            )
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Linear(in_channels if i == 0 else out_channels, out_channels)
                for i in range(num_layers + 1)
            ]
        )

    def forward(self, x, t_emb=None, context=None):
        out = x

        # First resnet block
        resnet_input = out
        out = self.mlp_first[0](out)
        if self.t_emb_dim is not None:
            out = out + self.t_emb_layers[0](t_emb)
        out = self.mlp_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)

        for i in range(self.num_layers):
            # Attention Block
            in_attn = self.attention_norms[i](out)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out = out + out_attn

            if self.cross_attn:
                assert context is not None
                in_attn = self.cross_attention_norms[i](out)
                context_proj = self.context_proj[i](context)
                out_attn, _ = self.cross_attentions[i](in_attn, context_proj, context_proj)
                out = out + out_attn

            # Resnet Block
            resnet_input = out
            out = self.mlp_first[i + 1](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i + 1](t_emb)
            out = self.mlp_second[i + 1](out)
            out = out + self.residual_input_conv[i + 1](resnet_input)

        return out


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim,
                 up_sample, num_heads, num_layers, attn, norm_channels):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample
        self.t_emb_dim = t_emb_dim
        self.attn = attn
        self.resnet_first = nn.ModuleList(
            [nn.Sequential(
                    nn.LayerNorm(in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Linear(in_channels if i == 0 else out_channels, out_channels),
                ) for i in range(num_layers)]
        )

        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                )
                for _ in range(num_layers)
            ])

        self.resnet_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(out_channels),
                    nn.SiLU(),
                    nn.Linear(out_channels, out_channels),
                )
                for _ in range(num_layers)
            ]
        )
        if self.attn:
            self.attention_norms = nn.ModuleList(
                [
                    nn.LayerNorm(out_channels)
                    for _ in range(num_layers)
                ]
            )

            self.attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                    for _ in range(num_layers)
                ]
            )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Linear(in_channels if i == 0 else out_channels, out_channels)
                for i in range(num_layers)
            ]
        )
        self.up_sample_conv = nn.Linear(in_channels, in_channels) \
            if self.up_sample else nn.Identity()

    def forward(self, x, out_down=None, t_emb=None):
        # Upsample
        x = self.up_sample_conv(x)

        if out_down is not None:
            x = torch.cat([x, out_down], dim=-1)

        out = x
        for i in range(self.num_layers):
            # Resnet Block
            resnet_input = out
            out = self.resnet_first[i](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i](t_emb)
            out = self.resnet_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)

            if self.attn:
                in_attn = self.attention_norms[i](out)
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                out = out + out_attn
        return out


class UpBlockUnet(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, up_sample,
                 num_heads, num_layers, cross_attn=False, context_dim=None):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample
        self.t_emb_dim = t_emb_dim
        self.cross_attn = cross_attn
        self.context_dim = context_dim
        self.resnet_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Linear(in_channels if i == 0 else out_channels, out_channels),
                )
                for i in range(num_layers)
            ]
        )

        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                )
                for _ in range(num_layers)
            ])

        self.resnet_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(out_channels),
                    nn.SiLU(),
                    nn.Linear(out_channels, out_channels),
                )
                for _ in range(num_layers)
            ]
        )

        self.attention_norms = nn.ModuleList(
            [
                nn.LayerNorm(out_channels)
                for _ in range(num_layers)
            ]
        )

        self.attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                for _ in range(num_layers)
            ]
        )

        if self.cross_attn:
            assert context_dim is not None
            self.cross_attention_norms = nn.ModuleList(
                [nn.LayerNorm(out_channels)
                 for _ in range(num_layers)]
            )
            self.cross_attentions = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                 for _ in range(num_layers)]
            )
            self.context_proj = nn.ModuleList(
                [nn.Linear(context_dim, out_channels)
                 for _ in range(num_layers)]
            )
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Linear(in_channels if i == 0 else out_channels, out_channels)
                for i in range(num_layers)
            ]
        )
        self.up_sample_conv = nn.Linear(in_channels, in_channels) \
            if self.up_sample else nn.Identity()

    def forward(self, x, out_down=None, t_emb=None, context=None):
        x = self.up_sample_conv(x)
        if out_down is not None:
            x = torch.cat([x, out_down], dim=-1)

        out = x
        for i in range(self.num_layers):
            # Resnet
            resnet_input = out
            out = self.resnet_first[i](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i](t_emb)
            out = self.resnet_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)

            in_attn = self.attention_norms[i](out)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out = out + out_attn

            if self.cross_attn:
                assert context is not None
                in_attn = self.cross_attention_norms[i](out)
                context_proj = self.context_proj[i](context)
                out_attn, _ = self.cross_attentions[i](in_attn, context_proj, context_proj)
                out = out + out_attn

        return out
