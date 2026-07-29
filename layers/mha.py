import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.dim import ensure_batch_first, restore_shape
from layers.Embed import absolute_sinusoidal_pe


class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, k_dim=None, v_dim=None, bias=True, batch_first=False):
        super().__init__()
        self.batch_first = batch_first

        if embed_dim % num_heads != 0:
            raise ValueError('embed_dim must be divisible by num_heads')

        self.k_dim = embed_dim // num_heads if k_dim is None else k_dim
        self.v_dim = embed_dim // num_heads if v_dim is None else v_dim
        self.num_heads = num_heads

        self.wq = nn.Linear(embed_dim, self.k_dim * self.num_heads, bias=bias)
        self.wk = nn.Linear(embed_dim, self.k_dim * self.num_heads, bias=bias)
        self.wv = nn.Linear(embed_dim, self.v_dim * self.num_heads, bias=bias)
        self.wo = nn.Linear(self.num_heads * self.v_dim, embed_dim, bias=bias)

    def split_heads(self, x, head_dim):
        b, t, c = x.size()
        return x.reshape(b, t, self.num_heads, head_dim).transpose(1, 2)

    def forward(self, q, k, v, attn_mask=None):
        dim_flags, q = ensure_batch_first(q, self.batch_first)
        _, k = ensure_batch_first(k, self.batch_first)
        _, v = ensure_batch_first(v, self.batch_first)

        q, k, v = self.wq(q), self.wk(k), self.wv(v)

        q = self.split_heads(q, self.k_dim)
        k = self.split_heads(k, self.k_dim)
        v = self.split_heads(v, self.v_dim)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        # attn = torch.matmul(F.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1]), dim=-1), v)  # same but slower
        attn = attn.transpose(1, 2).contiguous()
        attn = attn.view(attn.shape[0], attn.shape[1], -1)
        o = self.wo(attn)

        o = restore_shape(o, *dim_flags)
        return o


def _verify_mha_implementation():
    torch.manual_seed(47)
    embed_dim = 4
    num_heads = 2
    q = torch.rand(size=(1, 3, 4))
    x = torch.rand(size=(1, 3, 4))
    att = MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True, bias=False)

    if att.wq.bias is None:
        bias = None
    else:
        bias = torch.concat([att.wq.bias, att.wk.bias, att.wv.bias], dim=0)

    o = att(q, x, x)
    print(o)
    print(o.shape)
    o_val = F.multi_head_attention_forward(q.permute(1, 0, 2), x.permute(1, 0, 2), x.permute(1, 0, 2),
                                   num_heads=num_heads,
                                   training=False,
                                   add_zero_attn=False,
                                   embed_dim_to_check=embed_dim,
                                   dropout_p=0,
                                   in_proj_weight=None,
                                   in_proj_bias=bias,
                                   q_proj_weight=att.wq.weight,
                                   k_proj_weight=att.wk.weight,
                                   v_proj_weight=att.wv.weight,
                                   bias_k=None,
                                   bias_v=None,
                                   out_proj_weight=att.wo.weight,
                                   out_proj_bias=att.wo.bias,
                                   use_separate_proj_weight=True,
                                   need_weights=False)[0].permute(1, 0, 2)
    print(o_val)
    print(o_val.shape)
    print(torch.all(torch.isclose(o, o_val)))


class TUPEMultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, k_dim=None, v_dim=None, bias=True, batch_first=False, pe='sinusoidal', max_pe=5000):
        super().__init__()
        self.batch_first = batch_first

        if embed_dim % num_heads != 0:
            raise ValueError('embed_dim must be divisible by num_heads')

        self.k_dim = embed_dim // num_heads if k_dim is None else k_dim
        self.v_dim = embed_dim // num_heads if v_dim is None else v_dim
        self.num_heads = num_heads

        # input projections
        self.wq = nn.Linear(embed_dim, self.k_dim * self.num_heads, bias=bias)
        self.wk = nn.Linear(embed_dim, self.k_dim * self.num_heads, bias=bias)
        self.wv = nn.Linear(embed_dim, self.v_dim * self.num_heads, bias=bias)

        # positional projections
        self.wq_p = nn.Linear(embed_dim, self.k_dim * self.num_heads, bias=bias)
        self.wk_p = nn.Linear(embed_dim, self.k_dim * self.num_heads, bias=bias)

        self.wo = nn.Linear(self.num_heads * self.v_dim, embed_dim, bias=bias)

        if pe == 'sinusoidal':
            self.pos_enc = absolute_sinusoidal_pe(max_len=max_pe, d_model=embed_dim, batch_first=True)
        elif pe == 'learned':
            self.pos_enc = nn.Embedding(num_embeddings=max_pe, embedding_dim=embed_dim)
        else:
            raise ValueError(f'invalid positional encoding {pe}')

    def positional_encoding(self, x):
        if isinstance(self.pos_enc, torch.Tensor):
            pe = self.pos_enc[:, :x.shape[1], :]
        elif isinstance(self.pos_enc, nn.Embedding):
            pe = self.pos_enc(torch.arange(0, x.shape[1])).unsqueeze(dim=0)
        else:
            raise ValueError('invalid pe')

        return pe.repeat(x.shape[0], 1, 1).to(device=x.device)

    def split_heads(self, x, head_dim):
        b, t, c = x.size()
        return x.reshape(b, t, self.num_heads, head_dim).transpose(1, 2)

    def forward(self, x):
        dim_flags, x = ensure_batch_first(x, self.batch_first)
        x_shape = x.shape

        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        q = self.split_heads(q, self.k_dim)
        k = self.split_heads(k, self.k_dim)
        v = self.split_heads(v, self.v_dim)

        pe = self.positional_encoding(x)
        q_p, k_p = self.wq_p(pe), self.wk_p(pe)

        q_p = self.split_heads(q_p, self.k_dim)
        k_p = self.split_heads(k_p, self.k_dim)

        content = torch.matmul(q, k.transpose(-1, -2))
        position = torch.matmul(q_p, k_p.transpose(-1, -2))

        attn = torch.matmul(F.softmax((content + position) / math.sqrt(2 * self.k_dim), dim=-1), v)
        o = attn.transpose(1, 2).contiguous().view(x_shape[0], x_shape[1], self.num_heads * self.v_dim)
        o = self.wo(o)

        o = restore_shape(o, *dim_flags)
        return o


def _tupe_test():
    torch.manual_seed(47)
    x = torch.rand(size=(3, 31, 16))
    att = TUPEMultiheadAttention(embed_dim=x.shape[-1], num_heads=8, batch_first=True)
    o = att(x)
    print(o.shape)


def get_relative_dist(q_len, k_len, causal):
    q_pos = torch.arange(q_len).unsqueeze(1)
    k_pos = torch.arange(k_len).unsqueeze(0)

    if causal:
        rel_dist = (q_pos - k_pos).clamp(min=0)
    else:
        rel_dist = (q_pos - k_pos).abs()

    return rel_dist


def geometric_progression(start, ratio, n):
    return [start * ratio ** i for i in range(n)]


def get_slopes_power_of_2(n):
    start = (2 ** (-2 ** -(math.log2(n) - 3)))
    ratio = start
    return [start * ratio ** i for i in range(n)]


def get_slopes(n):
    if math.log2(n).is_integer():
        slopes = get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        slopes = get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

    return torch.tensor(slopes, dtype=torch.float32)


def get_alibi_mask(n_heads, q_len, k_len, causal):
    slopes = get_slopes(n_heads)
    rel_dist = get_relative_dist(q_len, k_len, causal)
    bias = -slopes[:, None, None] * rel_dist[None, :, :]
    return bias


class ALiBiMultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, k_dim=None, v_dim=None, bias=True, batch_first=False):
        super().__init__()
        self.batch_first = batch_first

        if embed_dim % num_heads != 0:
            raise ValueError('embed_dim must be divisible by num_heads')

        self.k_dim = embed_dim // num_heads if k_dim is None else k_dim
        self.v_dim = embed_dim // num_heads if v_dim is None else v_dim
        self.num_heads = num_heads

        self.wq = nn.Linear(embed_dim, self.k_dim * self.num_heads, bias=bias)
        self.wk = nn.Linear(embed_dim, self.k_dim * self.num_heads, bias=bias)
        self.wv = nn.Linear(embed_dim, self.v_dim * self.num_heads, bias=bias)
        self.wo = nn.Linear(self.num_heads * self.v_dim, embed_dim, bias=bias)

    def split_heads(self, x, head_dim):
        b, t, c = x.size()
        return x.reshape(b, t, self.num_heads, head_dim).transpose(1, 2)

    def forward(self, x, is_causal=False):
        dim_flags, x = ensure_batch_first(x, self.batch_first)
        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        q = self.split_heads(q, self.k_dim)
        k = self.split_heads(k, self.k_dim)
        v = self.split_heads(v, self.v_dim)

        attn_mask = get_alibi_mask(n_heads=self.num_heads, q_len=q.shape[2], k_len=k.shape[2], causal=is_causal)
        attn_mask = attn_mask.unsqueeze(dim=0).to(device=q.device)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn = attn.transpose(1, 2).contiguous()
        attn = attn.view(attn.shape[0], attn.shape[1], -1)
        o = self.wo(attn)

        o = restore_shape(o, *dim_flags)
        return o


def _alibi_test():
    torch.manual_seed(47)
    x = torch.rand(size=(3, 31, 16))
    att = ALiBiMultiheadAttention(embed_dim=x.shape[-1], num_heads=8, batch_first=True)
    o = att(x, is_causal=False)
    print(o.shape)


class MultiheadAttentionPooling(nn.Module):

    """MAP -> MultiheadAttentionPooling"""

    def __init__(self, num_seeds, embed_dim, num_heads=1, batch_first=False):
        super().__init__()
        self.batch_first = batch_first
        self.seed = nn.Parameter(torch.randn(size=(1, num_seeds, embed_dim)))
        nn.init.kaiming_normal_(self.seed)
        self.norm = nn.LayerNorm(embed_dim)
        self.pool = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        dim_flags, h = ensure_batch_first(x, self.batch_first)
        h = self.norm(h)
        seed = self.seed.expand(h.shape[0], -1, -1)
        h = self.pool(seed, h, h, need_weights=False)[0]
        h = restore_shape(h, *dim_flags)
        return h


def _mha_pooling():
    torch.manual_seed(47)
    x = torch.rand(size=(3, 31, 16))
    pooling = MultiheadAttentionPooling(num_seeds=2, embed_dim=x.shape[-1], num_heads=4, batch_first=True)
    o = pooling(x)
    print(o.shape)


class MultiheadAttentionMemory(nn.Module):

    def __init__(self, num_seeds, embed_dim, num_heads=1, batch_first=False):
        super().__init__()
        self.batch_first = batch_first
        self.keys = nn.Parameter(torch.randn(size=(1, num_seeds, embed_dim)))
        self.values = nn.Parameter(torch.randn(size=(1, num_seeds, embed_dim)))
        nn.init.kaiming_normal_(self.keys)
        nn.init.kaiming_normal_(self.values)
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        dim_flags, h = ensure_batch_first(x, self.batch_first)
        h = self.norm(h)
        keys = self.keys.expand(h.shape[0], -1, -1)
        values = self.values.expand(h.shape[0], -1, -1)
        h = h + self.attn(h, keys, values, need_weights=False)[0]
        h = restore_shape(h, *dim_flags)
        return h


def _mha_memory():
    torch.manual_seed(47)
    x = torch.rand(size=(3, 31, 4))
    pooling = MultiheadAttentionMemory(num_seeds=7, embed_dim=x.shape[-1], num_heads=2, batch_first=True)
    o = pooling(x)
    print(o.shape)


class TUPETransformerEncoder(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, bias=True, batch_first=False):
        super().__init__()

        self.self_attn = TUPEMultiheadAttention(embed_dim=d_model, num_heads=num_heads, bias=bias, batch_first=batch_first)
        self.norm1 = nn.LayerNorm(d_model, bias=bias)

        self.ffn1 = nn.Linear(in_features=d_model, out_features=d_ff, bias=bias)
        self.ffn2 = nn.Linear(in_features=d_ff, out_features=d_model, bias=bias)
        self.norm2 = nn.LayerNorm(d_model, bias=bias)

    def forward(self, x):
        h = x
        h = self.norm1(h + self.self_attn(h))
        h = self.norm2(h + self.ffn2(F.relu(self.ffn1(h))))
        return h


class ALiBiTransformerEncoder(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, bias=True, batch_first=False):
        super().__init__()

        self.self_attn = ALiBiMultiheadAttention(embed_dim=d_model, num_heads=num_heads, bias=bias, batch_first=batch_first)
        self.norm1 = nn.LayerNorm(d_model, bias=bias)

        self.ffn1 = nn.Linear(in_features=d_model, out_features=d_ff, bias=bias)
        self.ffn2 = nn.Linear(in_features=d_ff, out_features=d_model, bias=bias)
        self.norm2 = nn.LayerNorm(d_model, bias=bias)

    def forward(self, x):
        h = x
        h = self.norm1(h + self.self_attn(h))
        h = self.norm2(h + self.ffn2(F.relu(self.ffn1(h))))
        return h


if __name__ == '__main__':
    # _tupe_test()
    # _verify_mha_implementation()
    # _mha_pooling()
    # _mha_memory()
    _alibi_test()
