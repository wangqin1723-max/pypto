# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Minimal complete LLaMA 7B-style language model, using PyPTO language DSL.

Minimal single-head configuration:
  hidden_size = 64  (num_heads=1 × head_dim=64)
  num_heads   = 1
  head_dim    = 64
  head_dim/2  = 32  (for RoPE half-rotation)
  seq_len     = 16
  vocab_size  = 64  (reduced for hardware compatibility)

Architecture (same as llama_7b_full.py, simplified for single head):
  hidden [16,64]
    → Decoder Layer: RMSNorm → QKV[16,64] → RoPE → scaled Q@K^T → +mask
                    → softmax → @V → dense → residual →
                    → RMSNorm → SwiGLU MLP → residual
    → Final RMSNorm
    → LM Head: [16,64] @ [64,64] → logits [16,64]

Key simplification vs llama_7b_full.py:
  - Single head: no head splitting (kernel_extract_head) or concat needed
  - All tensors are [S, DH] = [16, 64] or [S, S] = [16, 16]

Hardware tile dimensions:
  S=16   sequence length
  D=64   hidden size (= head_dim for single-head)
  DH=64  head dimension
  DH2=32 half head dimension (for RoPE)
  V=64   vocabulary size

Reference: LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)
"""

import pypto.language as pl

# Hardware tile dimensions
_S = 16  # sequence length
_D = 64  # hidden size (num_heads × head_dim = 1 × 64)
_H = 1  # number of attention heads
_DH = 64  # head dimension
_DH2 = 32  # half head dimension (for RoPE)
_V = 64  # vocabulary size


@pl.program
class LlamaMiniProgram:
    """Minimal LLaMA 7B-style model: 1 decoder layer, 1 head, RoPE, LM head.

    Dimensions: hidden_size=64, num_heads=1, head_dim=64, seq_len=16, vocab_size=64.

    Architecture:
      hidden [16,64]
        → Decoder Layer: RMSNorm→QKV→RoPE→scaled-attn→dense→add
                       → RMSNorm→SwiGLU MLP→add
        → Final RMSNorm
        → LM Head: [16,64] @ [64,64] → logits [16,64]
    """

    # =========================================================================
    # InCore kernel: RMSNorm [16, 64]
    # Formula: x / sqrt(mean(x^2) + eps), divisor = hidden_size = 64
    # =========================================================================

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_rms_norm(
        self,
        x: pl.Tensor[[16, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
    ) -> pl.Tensor[[16, 64], pl.FP32]:
        """RMSNorm: x / sqrt(mean(x^2) + eps) across hidden_size=64."""
        tile_x: pl.Tile[[16, 64], pl.FP32] = pl.load(x, [0, 0], [16, 64], target_memory=pl.MemorySpace.Vec)

        squared: pl.Tile[[16, 64], pl.FP32] = pl.mul(tile_x, tile_x)

        tmp: pl.Tile[[16, 64], pl.FP32] = pl.create_tile(
            [16, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        mean_sq: pl.Tile[[16, 1], pl.FP32] = pl.row_sum(squared, tmp)
        # [16, 1] is ColMajor; reshape to [1, 16] for scalar mul, then back
        mean_sq_T: pl.Tile[[1, 16], pl.FP32] = pl.reshape(mean_sq, [1, 16])
        mean_sq_T = pl.mul(mean_sq_T, 0.015625)  # 1.0 / 64  # type: ignore[reportArgumentType]
        mean_sq = pl.reshape(mean_sq_T, [16, 1])

        mean_sq_T2: pl.Tile[[1, 16], pl.FP32] = pl.reshape(mean_sq, [1, 16])
        rms_T: pl.Tile[[1, 16], pl.FP32] = pl.add(mean_sq_T2, 1e-6)  # type: ignore[reportArgumentType]
        rms_T = pl.sqrt(rms_T)
        rms: pl.Tile[[16, 1], pl.FP32] = pl.reshape(rms_T, [16, 1])

        normalized: pl.Tile[[16, 64], pl.FP32] = pl.row_expand_div(tile_x, rms)
        out: pl.Tensor[[16, 64], pl.FP32] = pl.store(normalized, [0, 0], [16, 64], output)
        return out

    # =========================================================================
    # InCore kernel: matmul [16, 64] @ [64, 64] → [16, 64]
    # Used for QKV projections, dense projection, MLP projections, and LM head.
    # =========================================================================

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_matmul(
        self,
        a: pl.Tensor[[16, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
    ) -> pl.Tensor[[16, 64], pl.FP32]:
        """[16,64] @ [64,64] → [16,64] matrix multiplication."""
        tile_a_l1 = pl.load(a, [0, 0], [16, 64], target_memory=pl.MemorySpace.Mat)
        tile_b_l1 = pl.load(b, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
        tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
        tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
        tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
        out = pl.l0c_store(tile_c_l0c, [0, 0], [16, 64], output)
        return out

    # =========================================================================
    # InCore kernel: matmul with B transposed [16, 64] @ [16, 64]^T → [16, 16]
    # Used for Q @ K^T in single-head attention.
    # =========================================================================

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_matmul_trans_b(
        self,
        a: pl.Tensor[[16, 64], pl.FP32],
        b: pl.Tensor[[16, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """[16,64] @ [16,64]^T → [16,16]: Q @ K^T via K-tiled 16×16 matmul.

        Tiles K-dimension (64) into 4×16 square blocks so each TMOV operates on
        Mat[16,16] → Right[16,16], satisfying the hardware src.shape == dst.shape
        constraint. Direct TMOV of Mat[16,64] to Right[64,16] (transposed) violates
        this constraint on A2/A3 hardware.
        """
        # K-tile 0: columns [0:16]
        a0 = pl.load(a, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
        b0 = pl.load(b, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
        a0_l = pl.move(a0, target_memory=pl.MemorySpace.Left)
        b0_r = pl.move(b0, target_memory=pl.MemorySpace.Right, transpose=True)
        acc: pl.Tile[[16, 16], pl.FP32] = pl.matmul(a0_l, b0_r)

        # K-tile 1: columns [16:32]
        a1 = pl.load(a, [0, 16], [16, 16], target_memory=pl.MemorySpace.Mat)
        b1 = pl.load(b, [0, 16], [16, 16], target_memory=pl.MemorySpace.Mat)
        a1_l = pl.move(a1, target_memory=pl.MemorySpace.Left)
        b1_r = pl.move(b1, target_memory=pl.MemorySpace.Right, transpose=True)
        acc = pl.matmul_acc(acc, a1_l, b1_r)

        # K-tile 2: columns [32:48]
        a2 = pl.load(a, [0, 32], [16, 16], target_memory=pl.MemorySpace.Mat)
        b2 = pl.load(b, [0, 32], [16, 16], target_memory=pl.MemorySpace.Mat)
        a2_l = pl.move(a2, target_memory=pl.MemorySpace.Left)
        b2_r = pl.move(b2, target_memory=pl.MemorySpace.Right, transpose=True)
        acc = pl.matmul_acc(acc, a2_l, b2_r)

        # K-tile 3: columns [48:64]
        a3 = pl.load(a, [0, 48], [16, 16], target_memory=pl.MemorySpace.Mat)
        b3 = pl.load(b, [0, 48], [16, 16], target_memory=pl.MemorySpace.Mat)
        a3_l = pl.move(a3, target_memory=pl.MemorySpace.Left)
        b3_r = pl.move(b3, target_memory=pl.MemorySpace.Right, transpose=True)
        acc = pl.matmul_acc(acc, a3_l, b3_r)

        out = pl.l0c_store(acc, [0, 0], [16, 16], output)
        return out

    # =========================================================================
    # InCore kernel: attention output [16, 16] @ [16, 64] → [16, 64]
    # Used for probs @ V in single-head attention.
    # =========================================================================

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_matmul_attn(
        self,
        a: pl.Tensor[[16, 16], pl.FP32],
        b: pl.Tensor[[16, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
    ) -> pl.Tensor[[16, 64], pl.FP32]:
        """[16,16] @ [16,64] → [16,64]: probs @ V for single-head attention."""
        tile_a_l1 = pl.load(a, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
        tile_b_l1 = pl.load(b, [0, 0], [16, 64], target_memory=pl.MemorySpace.Mat)
        tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
        tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
        tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
        out = pl.l0c_store(tile_c_l0c, [0, 0], [16, 64], output)
        return out

    # =========================================================================
    # InCore kernel: row-wise softmax [16, 16]
    # =========================================================================

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_softmax(
        self,
        a: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Row-wise numerically stable softmax."""
        tile_a: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])

        max_tmp: pl.Tile[[16, 16], pl.FP32] = pl.create_tile(
            [16, 16], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        row_max: pl.Tile[[16, 1], pl.FP32] = pl.row_max(tile_a, max_tmp)

        shifted: pl.Tile[[16, 16], pl.FP32] = pl.row_expand_sub(tile_a, row_max)
        exp_shifted: pl.Tile[[16, 16], pl.FP32] = pl.exp(shifted)

        sum_tmp: pl.Tile[[16, 16], pl.FP32] = pl.create_tile(
            [16, 16], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        row_sum: pl.Tile[[16, 1], pl.FP32] = pl.row_sum(exp_shifted, sum_tmp)
        result: pl.Tile[[16, 16], pl.FP32] = pl.row_expand_div(exp_shifted, row_sum)

        out: pl.Tensor[[16, 16], pl.FP32] = pl.store(result, [0, 0], [16, 16], output)
        return out

    # =========================================================================
    # InCore kernel: scale attention scores [16, 16] × 0.125
    # 1/sqrt(head_dim=64) = 0.125
    # =========================================================================

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_scale_scores(
        self,
        scores: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Scale attention scores by 1/sqrt(head_dim=64) = 0.125."""
        tile: pl.Tile[[16, 16], pl.FP32] = pl.load(scores, [0, 0], [16, 16])
        scaled: pl.Tile[[16, 16], pl.FP32] = pl.mul(tile, 0.125)  # type: ignore[reportArgumentType]
        out: pl.Tensor[[16, 16], pl.FP32] = pl.store(scaled, [0, 0], [16, 16], output)
        return out

    # =========================================================================
    # InCore kernel: element-wise add [16, 16]
    # Used for applying causal mask to attention scores.
    # =========================================================================

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_add_scores(
        self,
        a: pl.Tensor[[16, 16], pl.FP32],
        b: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        """Element-wise addition [16,16]: output = a + b."""
        tile_a: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
        tile_b: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
        result: pl.Tile[[16, 16], pl.FP32] = pl.add(tile_a, tile_b)
        out: pl.Tensor[[16, 16], pl.FP32] = pl.store(result, [0, 0], [16, 16], output)
        return out

    # =========================================================================
    # InCore kernel: element-wise add [16, 64]
    # Used for residual connections.
    # =========================================================================

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_add(
        self,
        a: pl.Tensor[[16, 64], pl.FP32],
        b: pl.Tensor[[16, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
    ) -> pl.Tensor[[16, 64], pl.FP32]:
        """Element-wise addition [16,64]: output = a + b."""
        tile_a: pl.Tile[[16, 64], pl.FP32] = pl.load(a, [0, 0], [16, 64], target_memory=pl.MemorySpace.Vec)
        tile_b: pl.Tile[[16, 64], pl.FP32] = pl.load(b, [0, 0], [16, 64], target_memory=pl.MemorySpace.Vec)
        result: pl.Tile[[16, 64], pl.FP32] = pl.add(tile_a, tile_b)
        out: pl.Tensor[[16, 64], pl.FP32] = pl.store(result, [0, 0], [16, 64], output)
        return out

    # =========================================================================
    # InCore kernel: RoPE [16, 64] using [16, 32] cos/sin tables
    # Rotate-half pattern: same as llama_7b_full.py with S=16, single head.
    # =========================================================================

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_rope(
        self,
        x: pl.Tensor[[16, 64], pl.FP32],
        cos_emb: pl.Tensor[[16, 32], pl.FP32],
        sin_emb: pl.Tensor[[16, 32], pl.FP32],
        output: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
    ) -> pl.Tensor[[16, 64], pl.FP32]:
        """Apply Rotary Position Embedding to a [S, head_dim] tensor.

        Rotate-half RoPE:
          x_left  = x[:, :32]   rotated_left  = x_left * cos - x_right * sin
          x_right = x[:, 32:]   rotated_right = x_right * cos + x_left * sin
        """
        x_left: pl.Tile[[16, 32], pl.FP32] = pl.load(x, [0, 0], [16, 32])
        x_right: pl.Tile[[16, 32], pl.FP32] = pl.load(x, [0, 32], [16, 32])
        cos_tile: pl.Tile[[16, 32], pl.FP32] = pl.load(cos_emb, [0, 0], [16, 32])
        sin_tile: pl.Tile[[16, 32], pl.FP32] = pl.load(sin_emb, [0, 0], [16, 32])

        left_cos: pl.Tile[[16, 32], pl.FP32] = pl.mul(x_left, cos_tile)
        right_sin: pl.Tile[[16, 32], pl.FP32] = pl.mul(x_right, sin_tile)
        rotated_left: pl.Tile[[16, 32], pl.FP32] = pl.sub(left_cos, right_sin)

        right_cos: pl.Tile[[16, 32], pl.FP32] = pl.mul(x_right, cos_tile)
        left_sin: pl.Tile[[16, 32], pl.FP32] = pl.mul(x_left, sin_tile)
        rotated_right: pl.Tile[[16, 32], pl.FP32] = pl.add(right_cos, left_sin)

        out_left: pl.Tensor[[16, 64], pl.FP32] = pl.store(rotated_left, [0, 0], [16, 32], output)
        out: pl.Tensor[[16, 64], pl.FP32] = pl.store(rotated_right, [0, 32], [16, 32], out_left)
        return out

    # =========================================================================
    # InCore kernel: SwiGLU [16, 64]
    # Formula: SiLU(gate) * up = gate * sigmoid(gate) * up
    # =========================================================================

    @pl.function(type=pl.FunctionType.InCore)
    def kernel_swiglu(
        self,
        gate: pl.Tensor[[16, 64], pl.FP32],
        up: pl.Tensor[[16, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
    ) -> pl.Tensor[[16, 64], pl.FP32]:
        """SwiGLU: SiLU(gate) * up = gate * sigmoid(gate) * up."""
        tile_gate: pl.Tile[[16, 64], pl.FP32] = pl.load(
            gate, [0, 0], [16, 64], target_memory=pl.MemorySpace.Vec
        )
        tile_up: pl.Tile[[16, 64], pl.FP32] = pl.load(up, [0, 0], [16, 64], target_memory=pl.MemorySpace.Vec)

        gate_neg: pl.Tile[[16, 64], pl.FP32] = pl.mul(tile_gate, -1.0)  # type: ignore[reportArgumentType]
        exp_neg: pl.Tile[[16, 64], pl.FP32] = pl.exp(gate_neg)
        denom: pl.Tile[[16, 64], pl.FP32] = pl.add(exp_neg, 1.0)  # type: ignore[reportArgumentType]
        sigmoid: pl.Tile[[16, 64], pl.FP32] = pl.recip(denom)
        swish: pl.Tile[[16, 64], pl.FP32] = pl.mul(tile_gate, sigmoid)
        result: pl.Tile[[16, 64], pl.FP32] = pl.mul(swish, tile_up)

        out: pl.Tensor[[16, 64], pl.FP32] = pl.store(result, [0, 0], [16, 64], output)
        return out

    # =========================================================================
    # Top-level orchestration: minimal 1-layer, 1-head LLaMA 7B model
    # =========================================================================

    @pl.function(type=pl.FunctionType.Orchestration)
    def llama_mini_orch(
        self,
        # Input hidden states
        hidden: pl.Tensor[[16, 64], pl.FP32],
        # Causal attention mask
        causal_mask: pl.Tensor[[16, 16], pl.FP32],
        # RoPE embeddings: positions × head_dim/2
        cos_emb: pl.Tensor[[16, 32], pl.FP32],
        sin_emb: pl.Tensor[[16, 32], pl.FP32],
        # QKV and projection weights [hidden, hidden] = [64, 64]
        wq: pl.Tensor[[64, 64], pl.FP32],
        wk: pl.Tensor[[64, 64], pl.FP32],
        wv: pl.Tensor[[64, 64], pl.FP32],
        w_dense: pl.Tensor[[64, 64], pl.FP32],
        # MLP weights [hidden, hidden] = [64, 64]
        w_gate: pl.Tensor[[64, 64], pl.FP32],
        w_up: pl.Tensor[[64, 64], pl.FP32],
        w_down: pl.Tensor[[64, 64], pl.FP32],
        # LM head weight [hidden, vocab] = [64, 64]
        w_lm: pl.Tensor[[64, 64], pl.FP32],
    ) -> pl.Tensor[[16, 64], pl.FP32]:
        """Minimal LLaMA 7B-style model forward pass (1 layer, 1 head).

        Pipeline:
          hidden [16,64]
            → Decoder Layer:
                RMSNorm → QKV[16,64] → RoPE →
                scaled Q@K^T[16,16] → +mask → softmax → @V[16,64] →
                dense → residual →
                RMSNorm → SwiGLU MLP → residual
            → Final RMSNorm
            → LM Head: [16,64] @ [64,64] → logits [16,64]
        """
        # ===== Decoder Layer =====

        # Pre-attention RMSNorm
        normed: pl.Tensor[[16, 64], pl.FP32] = pl.create_tensor([16, 64], dtype=pl.FP32)
        normed = self.kernel_rms_norm(hidden, normed)

        # QKV projections: [16,64] @ [64,64] → [16,64] each
        q: pl.Tensor[[16, 64], pl.FP32] = pl.create_tensor([16, 64], dtype=pl.FP32)
        q = self.kernel_matmul(normed, wq, q)
        k: pl.Tensor[[16, 64], pl.FP32] = pl.create_tensor([16, 64], dtype=pl.FP32)
        k = self.kernel_matmul(normed, wk, k)
        v: pl.Tensor[[16, 64], pl.FP32] = pl.create_tensor([16, 64], dtype=pl.FP32)
        v = self.kernel_matmul(normed, wv, v)

        # Apply RoPE to Q and K
        q_rot: pl.Tensor[[16, 64], pl.FP32] = pl.create_tensor([16, 64], dtype=pl.FP32)
        q_rot = self.kernel_rope(q, cos_emb, sin_emb, q_rot)
        k_rot: pl.Tensor[[16, 64], pl.FP32] = pl.create_tensor([16, 64], dtype=pl.FP32)
        k_rot = self.kernel_rope(k, cos_emb, sin_emb, k_rot)

        # Scaled causal dot-product attention
        # scores = Q @ K^T → [16, 16]
        scores: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
        scores = self.kernel_matmul_trans_b(q_rot, k_rot, scores)
        # Scale by 1/sqrt(head_dim=64) = 0.125
        scaled: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
        scaled = self.kernel_scale_scores(scores, scaled)
        # Apply causal mask
        masked: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
        masked = self.kernel_add_scores(scaled, causal_mask, masked)
        # Softmax
        probs: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
        probs = self.kernel_softmax(masked, probs)
        # probs @ V → [16, 64]
        attn_out: pl.Tensor[[16, 64], pl.FP32] = pl.create_tensor([16, 64], dtype=pl.FP32)
        attn_out = self.kernel_matmul_attn(probs, v, attn_out)

        # Dense projection + first residual
        dense_out: pl.Tensor[[16, 64], pl.FP32] = pl.create_tensor([16, 64], dtype=pl.FP32)
        dense_out = self.kernel_matmul(attn_out, w_dense, dense_out)
        attn_res: pl.Tensor[[16, 64], pl.FP32] = pl.create_tensor([16, 64], dtype=pl.FP32)
        attn_res = self.kernel_add(hidden, dense_out, attn_res)

        # Pre-MLP RMSNorm
        normed2: pl.Tensor[[16, 64], pl.FP32] = pl.create_tensor([16, 64], dtype=pl.FP32)
        normed2 = self.kernel_rms_norm(attn_res, normed2)

        # SwiGLU MLP
        gate: pl.Tensor[[16, 64], pl.FP32] = pl.create_tensor([16, 64], dtype=pl.FP32)
        gate = self.kernel_matmul(normed2, w_gate, gate)
        up: pl.Tensor[[16, 64], pl.FP32] = pl.create_tensor([16, 64], dtype=pl.FP32)
        up = self.kernel_matmul(normed2, w_up, up)
        swish_up: pl.Tensor[[16, 64], pl.FP32] = pl.create_tensor([16, 64], dtype=pl.FP32)
        swish_up = self.kernel_swiglu(gate, up, swish_up)
        mlp_out: pl.Tensor[[16, 64], pl.FP32] = pl.create_tensor([16, 64], dtype=pl.FP32)
        mlp_out = self.kernel_matmul(swish_up, w_down, mlp_out)

        # Second residual → h1
        h1: pl.Tensor[[16, 64], pl.FP32] = pl.create_tensor([16, 64], dtype=pl.FP32)
        h1 = self.kernel_add(attn_res, mlp_out, h1)

        # ===== Final RMSNorm =====
        h_normed: pl.Tensor[[16, 64], pl.FP32] = pl.create_tensor([16, 64], dtype=pl.FP32)
        h_normed = self.kernel_rms_norm(h1, h_normed)

        # ===== LM Head: [16,64] @ [64,64] → logits [16,64] =====
        logits: pl.Tensor[[16, 64], pl.FP32] = pl.create_tensor([16, 64], dtype=pl.FP32)
        logits = self.kernel_matmul(h_normed, w_lm, logits)

        return logits
