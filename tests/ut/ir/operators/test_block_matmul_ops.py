# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Unit tests for block-level matrix multiplication operators."""

import pypto.language as pl
from pypto import DataType, ir


class TestBlockMatMulOps:
    """Test suite for block-level matrix multiplication operators."""

    def test_block_matmul(self):
        """Test block.matmul operator - matrix multiplication."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 64], pl.FP32],
                b: pl.Tensor[[64, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 16], pl.FP32] = pl.load(a, [0, 0], [32, 16])
                tile_b: pl.Tile[[16, 32], pl.FP32] = pl.load(b, [0, 0], [16, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.matmul(tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.matmul" in ir_str

    def test_block_matmul_acc(self):
        """Test block.matmul_acc operator - matrix multiplication with accumulation (TMATMUL_ACC).

        Computes: acc_out = acc_in + lhs @ rhs
        """

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                acc_in: pl.Tensor[[128, 128], pl.FP32],
                a: pl.Tensor[[128, 64], pl.FP32],
                b: pl.Tensor[[64, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_acc: pl.Tile[[32, 32], pl.FP32] = pl.load(acc_in, [0, 0], [32, 32])
                tile_a: pl.Tile[[32, 16], pl.FP32] = pl.load(a, [0, 0], [32, 16])
                tile_b: pl.Tile[[16, 32], pl.FP32] = pl.load(b, [0, 0], [16, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.matmul_acc(tile_acc, tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.matmul_acc" in ir_str

    def test_block_matmul_bias(self):
        """Test block.matmul_bias operator - matrix multiplication with bias add."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 64], pl.FP32],
                b: pl.Tensor[[64, 128], pl.FP32],
                bias: pl.Tensor[[1, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 16], pl.FP32] = pl.load(a, [0, 0], [32, 16])
                tile_b: pl.Tile[[16, 32], pl.FP32] = pl.load(b, [0, 0], [16, 32])
                tile_bias: pl.Tile[[1, 32], pl.FP32] = pl.load(bias, [0, 0], [1, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.matmul_bias(tile_a, tile_b, tile_bias)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.matmul_bias" in ir_str

    def test_block_gemv(self):
        """Test block.gemv operator - general matrix-vector multiplication."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[1, 64], pl.FP32],
                b: pl.Tensor[[64, 128], pl.FP32],
                output: pl.Tensor[[1, 128], pl.FP32],
            ) -> pl.Tensor[[1, 128], pl.FP32]:
                tile_a: pl.Tile[[1, 16], pl.FP32] = pl.load(a, [0, 0], [1, 16])
                tile_b: pl.Tile[[16, 32], pl.FP32] = pl.load(b, [0, 0], [16, 32])
                tile_c: pl.Tile[[1, 32], pl.FP32] = pl.gemv(tile_a, tile_b)
                result: pl.Tensor[[1, 128], pl.FP32] = pl.store(tile_c, [0, 0], [1, 32], output)
                return result

        ir_str = str(Program)
        assert "block.gemv" in ir_str

    def test_block_gemv_acc(self):
        """Test block.gemv_acc operator - GEMV with accumulation."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                acc_in: pl.Tensor[[1, 128], pl.FP32],
                a: pl.Tensor[[1, 64], pl.FP32],
                b: pl.Tensor[[64, 128], pl.FP32],
                output: pl.Tensor[[1, 128], pl.FP32],
            ) -> pl.Tensor[[1, 128], pl.FP32]:
                tile_acc: pl.Tile[[1, 32], pl.FP32] = pl.load(acc_in, [0, 0], [1, 32])
                tile_a: pl.Tile[[1, 16], pl.FP32] = pl.load(a, [0, 0], [1, 16])
                tile_b: pl.Tile[[16, 32], pl.FP32] = pl.load(b, [0, 0], [16, 32])
                tile_c: pl.Tile[[1, 32], pl.FP32] = pl.gemv_acc(tile_acc, tile_a, tile_b)
                result: pl.Tensor[[1, 128], pl.FP32] = pl.store(tile_c, [0, 0], [1, 32], output)
                return result

        ir_str = str(Program)
        assert "block.gemv_acc" in ir_str

    def test_block_gemv_bias(self):
        """Test block.gemv_bias operator - GEMV with bias add."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[1, 64], pl.FP32],
                b: pl.Tensor[[64, 128], pl.FP32],
                bias: pl.Tensor[[1, 128], pl.FP32],
                output: pl.Tensor[[1, 128], pl.FP32],
            ) -> pl.Tensor[[1, 128], pl.FP32]:
                tile_a: pl.Tile[[1, 16], pl.FP32] = pl.load(a, [0, 0], [1, 16])
                tile_b: pl.Tile[[16, 32], pl.FP32] = pl.load(b, [0, 0], [16, 32])
                tile_bias: pl.Tile[[1, 32], pl.FP32] = pl.load(bias, [0, 0], [1, 32])
                tile_c: pl.Tile[[1, 32], pl.FP32] = pl.gemv_bias(tile_a, tile_b, tile_bias)
                result: pl.Tensor[[1, 128], pl.FP32] = pl.store(tile_c, [0, 0], [1, 32], output)
                return result

        ir_str = str(Program)
        assert "block.gemv_bias" in ir_str


class TestBlockBatchMatMulOps:
    """Tests for block batch matrix multiplication operations."""

    def test_batch_matmul_2d(self):
        """Test block.batch_matmul with 2D tiles (equivalent to regular matmul)."""
        span = ir.Span.unknown()

        # Create 2D tiles: [16, 32] @ [32, 64] -> [16, 64]
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        dim32 = ir.ConstInt(32, DataType.INT32, span)
        dim64 = ir.ConstInt(64, DataType.INT32, span)

        lhs_type = ir.TileType([dim16, dim32], DataType.FP16)
        rhs_type = ir.TileType([dim32, dim64], DataType.FP16)

        lhs = ir.Var("lhs", lhs_type, span)
        rhs = ir.Var("rhs", rhs_type, span)

        # Create batch_matmul call
        call = ir.create_op_call("block.batch_matmul", [lhs, rhs], {}, span)

        assert isinstance(call, ir.Call)
        assert call.op.name == "block.batch_matmul"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert len(result_type.shape) == 2
        assert result_type.dtype == DataType.FP16

    def test_batch_matmul_3d(self):
        """Test block.batch_matmul with 3D tiles (batch dimension)."""
        span = ir.Span.unknown()

        # Create 3D tiles: [4, 16, 32] @ [4, 32, 64] -> [4, 16, 64]
        dim4 = ir.ConstInt(4, DataType.INT32, span)
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        dim32 = ir.ConstInt(32, DataType.INT32, span)
        dim64 = ir.ConstInt(64, DataType.INT32, span)

        lhs_type = ir.TileType([dim4, dim16, dim32], DataType.FP32)
        rhs_type = ir.TileType([dim4, dim32, dim64], DataType.FP32)

        lhs = ir.Var("lhs", lhs_type, span)
        rhs = ir.Var("rhs", rhs_type, span)

        # Create batch_matmul call
        call = ir.create_op_call("block.batch_matmul", [lhs, rhs], {}, span)

        assert isinstance(call, ir.Call)
        assert call.op.name == "block.batch_matmul"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert len(result_type.shape) == 3
        assert result_type.dtype == DataType.FP32

    def test_batch_matmul_4d(self):
        """Test block.batch_matmul with 4D tiles (multiple batch dimensions)."""
        span = ir.Span.unknown()

        # Create 4D tiles: [2, 3, 16, 32] @ [2, 3, 32, 64] -> [2, 3, 16, 64]
        dim2 = ir.ConstInt(2, DataType.INT32, span)
        dim3 = ir.ConstInt(3, DataType.INT32, span)
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        dim32 = ir.ConstInt(32, DataType.INT32, span)
        dim64 = ir.ConstInt(64, DataType.INT32, span)

        lhs_type = ir.TileType([dim2, dim3, dim16, dim32], DataType.FP16)
        rhs_type = ir.TileType([dim2, dim3, dim32, dim64], DataType.FP16)

        lhs = ir.Var("lhs", lhs_type, span)
        rhs = ir.Var("rhs", rhs_type, span)

        # Create batch_matmul call
        call = ir.create_op_call("block.batch_matmul", [lhs, rhs], {}, span)

        assert isinstance(call, ir.Call)
        assert call.op.name == "block.batch_matmul"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert len(result_type.shape) == 4
        assert result_type.dtype == DataType.FP16

    def test_batch_matmul_broadcast(self):
        """Test block.batch_matmul with broadcasting batch dimensions."""
        span = ir.Span.unknown()

        # Create tiles with different batch shapes: [1, 16, 32] @ [4, 32, 64] -> [4, 16, 64]
        dim1 = ir.ConstInt(1, DataType.INT32, span)
        dim4 = ir.ConstInt(4, DataType.INT32, span)
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        dim32 = ir.ConstInt(32, DataType.INT32, span)
        dim64 = ir.ConstInt(64, DataType.INT32, span)

        lhs_type = ir.TileType([dim1, dim16, dim32], DataType.FP32)
        rhs_type = ir.TileType([dim4, dim32, dim64], DataType.FP32)

        lhs = ir.Var("lhs", lhs_type, span)
        rhs = ir.Var("rhs", rhs_type, span)

        # Create batch_matmul call
        call = ir.create_op_call("block.batch_matmul", [lhs, rhs], {}, span)

        assert isinstance(call, ir.Call)
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert len(result_type.shape) == 3
