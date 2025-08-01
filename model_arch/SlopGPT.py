import torch
import torch.nn as nn
from typing import Dict, Optional, List
from model_arch.model_helpers import STR_TO_FN_MAP, SwiGLU
from utils.consts import (
    MLP_EXPANSION_FACTOR,
    DEFAULT_PAD_TOKEN_ID,
    DEFAULT_BOS_TOKEN_ID,
    DEFAULT_EOS_TOKEN_ID,
    DEFAULT_OOS_TOKEN_ID,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_NUM_HEADS,
    DEFAULT_NUM_LAYERS,
    DEFAULT_VOCAB_SIZE,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_RMS_NORM_EPS,
)
import math


class SlopGPTConfig:
    "learnable params based on jamba's ones, just a company I like. kv heads are gone, since its dense attn. moe stuff gone too. kv cache kept though"

    def __init__(self, **init_params):
        self.init_attn_dropout = init_params.get("attention_dropout", 0.0)
        self.pad_token_idx = init_params.get("pad_token_id", DEFAULT_PAD_TOKEN_ID)
        self.bos_token_idx = init_params.get("bos_token_id", DEFAULT_BOS_TOKEN_ID)
        self.eos_token_idx = init_params.get("eos_token_id", DEFAULT_EOS_TOKEN_ID)
        self.oos_token_idx = init_params.get("oos_token_id", DEFAULT_OOS_TOKEN_ID)
        self.rms_norm_eps = init_params.get("rms_norm_eps", DEFAULT_RMS_NORM_EPS)
        self.hidden_act_fn = STR_TO_FN_MAP.get(init_params.get("hidden_act", "swiglu"))
        self.hidden_size = init_params.get("hidden_size", DEFAULT_HIDDEN_SIZE)
        self.n_heads = init_params.get("num_attention_heads", DEFAULT_NUM_HEADS)
        self.n_h_layers = init_params.get("num_hidden_layers", DEFAULT_NUM_LAYERS)
        self.tie_word_embeddings = init_params.get("tie_word_embeddings", True)
        self.torch_dtype = init_params.get("torch_dtype", "bfloat16")
        self.use_kvcache = init_params.get("use_cache", True)
        self.vocab_size = init_params.get("vocab_size", DEFAULT_VOCAB_SIZE)
        self.block_size = init_params.get("block_size", DEFAULT_BLOCK_SIZE)


class SlopGPT(nn.Module):
    def __init__(self, config: SlopGPTConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_enc = nn.Parameter(
            torch.zeros(1, config.block_size, config.hidden_size)
        )  # learnable instead of [sin (k/n^(2i/d))], [cos(k/n^(2i/d))]

        self.layers = nn.ModuleList(
            [SlopGPTransformerBlock(self.config) for _ in range(config.n_h_layers)]
        )
        self.final_norm = nn.modules.normalization.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight

    def forward(
        self,
        input_idxs,
        use_cache: bool = False,
        kv_cache: Optional[List[Dict[str, torch.Tensor]]] = None,
    ):
        batch, seq_len = input_idxs.size()
        device = input_idxs.device  # TODO: lowkey a little freaky if I do multigpu
        new_kv_caches = []
        # pos = torch.arange(0, seq_len, device = device).unsqueeze(0)
        x = (
            self.embed(input_idxs) + self.pos_enc[:, :seq_len, :]
        )  # since we are only targeting position in the tensor
        mask = (
            torch.tril(torch.ones(seq_len, seq_len, device=device))
            .unsqueeze(0)
            .unsqueeze(0)
        )  # tensor of [[[1]], [1]], [seq_len]], [seq_len]]] to match shape of other tensor

        if use_cache and kv_cache:
            for i, layer in enumerate(self.layers):
                cache_i = kv_cache[i] if i < len(kv_cache) else None
                x, layer_kv = layer(x, mask, kv_cache=cache_i, use_cache=use_cache)
                new_kv_caches.append(layer_kv)
        else:
            for layer in self.layers:
                x, _ = layer(x, mask, kv_cache=None, use_cache=False)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits, new_kv_caches if use_cache else None


class SlopGPTransformerBlock(nn.Module):
    def __init__(self, config: SlopGPTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.n_heads
        assert self.hidden_size % self.n_heads == 0, "needs to be divisible"

        self.rmsnorm1 = nn.modules.normalization.RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )

        self.lin_proj_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.lin_proj_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.lin_proj_v = nn.Linear(self.hidden_size, self.hidden_size)

        self.lin_proj_o = nn.Linear(self.hidden_size, self.hidden_size)

        self.dropout = nn.Dropout(config.init_attn_dropout)
        self.rmsnorm_pre_res_conn = nn.modules.normalization.RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.residual_conn_1st_layer = nn.Linear(
            self.hidden_size, MLP_EXPANSION_FACTOR * self.hidden_size
        )
        if config.hidden_act_fn is not None:
            if config.hidden_act_fn == SwiGLU:
                self.residual_conn_act_fn = config.hidden_act_fn(
                    MLP_EXPANSION_FACTOR * self.hidden_size
                )
            else:
                self.residual_conn_act_fn = config.hidden_act_fn()
        else:
            self.residual_conn_act_fn = SwiGLU(MLP_EXPANSION_FACTOR * self.hidden_size)
        self.residual_conn_2nd_layer = nn.Linear(
            MLP_EXPANSION_FACTOR * self.hidden_size, self.hidden_size
        )
        self.residual_conn_dropout = nn.Dropout(config.init_attn_dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        use_cache: bool = False,
        verbose: bool = False,
    ):
        new_kv_cache = None
        batch, seq_len, hidden_dim = x.size()

        x_norm_pre_qkv = self.rmsnorm1(x)

        # for q, k, and v initially: 4 tensor of dimensions : [[[batch]][sequence_length]][[n_heads]][[head_dim]]], swapping dims 1 and 2 (since we're 0 indexed) ->
        # [[[batch]][n_heads]][[sequence_length]][[head_dim]]]

        q = (
            self.lin_proj_q(x_norm_pre_qkv)
            .view(batch, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.lin_proj_k(x_norm_pre_qkv)
            .view(batch, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.lin_proj_v(x_norm_pre_qkv)
            .view(batch, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )

        if kv_cache is not None:
            k = torch.cat([kv_cache["k"], k], dim=2)  # concat along seq_len dim
            v = torch.cat([kv_cache["v"], v], dim=2)  # look up brodigy ^

        if use_cache:
            new_kv_cache = {"k": k, "v": v}

        # whoa dude is that the famous vaswani et al 2017
        pre_mask_attn_scores = (q @ k.transpose(-2, -1)) / (math.sqrt(self.head_dim))
        masked_attn = pre_mask_attn_scores.masked_fill(mask == 0, float("-inf"))
        attn_probs = torch.softmax(masked_attn, dim=-1)
        attn_out = (
            (attn_probs @ v)
            .transpose(1, 2)
            .contiguous()
            .view(batch, seq_len, hidden_dim)
        )

        attn_x = x + self.dropout(self.lin_proj_o(attn_out))

        res_norm_x = self.rmsnorm_pre_res_conn(attn_x)
        res_norm_x = self.residual_conn_1st_layer(res_norm_x)
        res_norm_x = self.residual_conn_act_fn(res_norm_x)
        res_norm_x = self.residual_conn_2nd_layer(res_norm_x)
        res_norm_x = self.residual_conn_dropout(res_norm_x)
        out_x = attn_x + res_norm_x

        return out_x, new_kv_cache
