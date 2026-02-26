# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Unit tests for block-level transform operators."""

import pypto.language as pl
from pypto import DataType, ir
from pypto.ir.op import block

class TestBlockTransformOps:
    """Test suite for block-level transform operators."""

    def test_block_transpose(self):
        """Test block.transpose operator - transpose a tile."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 64], pl.FP32],
                output: pl.Tensor[[64, 128], pl.FP32],
            ) -> pl.Tensor[[64, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 16], pl.FP32] = pl.load(a, [0, 0], [32, 16])
                tile_c: pl.Tile[[16, 32], pl.FP32] = pl.transpose(tile_a, axis1=0, axis2=1)
                result: pl.Tensor[[64, 128], pl.FP32] = pl.store(tile_c, [0, 0], [16, 32], output)
                return result

        ir_str = str(Program)
        assert "block.transpose" in ir_str


class TestTileTransformOps:
    """Tests for tile transform operations."""

    def test_tile_view(self):
        """Test tile.view operation."""
        span = ir.Span.unknown()

        # Create a tile variable [16, 32]
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        dim32 = ir.ConstInt(32, DataType.INT32, span)
        tile_type = ir.TileType([dim16, dim32], DataType.FP16)
        tile_var = ir.Var("tile", tile_type, span)

        # Create a view [8, 16] with offset [0, 0]
        call = block.view(tile_var, [8, 16], [0, 0])

        assert isinstance(call, ir.Call)
        assert call.op.name == "block.view"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == DataType.FP16
        assert len(result_type.shape) == 2

    def test_tile_reshape(self):
        """Test tile.reshape operation."""
        span = ir.Span.unknown()

        # Create a tile variable [4, 8]
        dim4 = ir.ConstInt(4, DataType.INT32, span)
        dim8 = ir.ConstInt(8, DataType.INT32, span)
        tile_type = ir.TileType([dim4, dim8], DataType.FP32)
        tile_var = ir.Var("tile", tile_type, span)

        # Reshape to [8, 4]
        call = block.reshape(tile_var, [8, 4])

        assert isinstance(call, ir.Call)
        assert call.op.name == "block.reshape"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == DataType.FP32
        assert len(result_type.shape) == 2

        # Reshape to [32, 1]
        call2 = block.reshape(tile_var, [32, 1])
        result_type2 = call2.type
        assert isinstance(result_type2, ir.TileType)
        assert len(result_type2.shape) == 2

    def test_tile_transpose(self):
        """Test tile.transpose operation."""
        span = ir.Span.unknown()

        # Create a tile [8, 16]
        dim8 = ir.ConstInt(8, DataType.INT32, span)
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        tile_type = ir.TileType([dim8, dim16], DataType.FP16)
        tile_var = ir.Var("tile", tile_type, span)

        # Transpose: [8, 16] -> [16, 8]
        call = block.transpose(tile_var, 0, 1)

        assert isinstance(call, ir.Call)
        assert call.op.name == "block.transpose"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == DataType.FP16
        assert len(result_type.shape) == 2

    def test_tile_transpose_negative_axis(self):
        """Test tile.transpose with negative axis indices."""
        span = ir.Span.unknown()

        # Create a tile [8, 16]
        dim8 = ir.ConstInt(8, DataType.INT32, span)
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        tile_type = ir.TileType([dim8, dim16], DataType.FP32)
        tile_var = ir.Var("tile", tile_type, span)

        # Transpose using negative indices: axis1=-2 (0), axis2=-1 (1)
        # [8, 16] -> [16, 8]
        call = block.transpose(tile_var, -2, -1)

        assert isinstance(call, ir.Call)
        assert call.op.name == "block.transpose"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)

    def test_transform_operators_registered(self):
        """Test that transform operators are registered."""
        assert ir.is_op_registered("block.view")
        assert ir.is_op_registered("block.reshape")
        assert ir.is_op_registered("block.transpose")

class TestMultiDimensionalTileOps:
    """Tests for multi-dimensional TileType operations."""

    def test_transpose_3d(self):
        """Test transpose on 3D tile."""
        span = ir.Span.unknown()

        # Create a 3D tile [4, 8, 16]
        dim4 = ir.ConstInt(4, DataType.INT32, span)
        dim8 = ir.ConstInt(8, DataType.INT32, span)
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        tile_type = ir.TileType([dim4, dim8, dim16], DataType.FP16)
        tile_var = ir.Var("tile", tile_type, span)

        # Transpose axes 0 and 2: [4, 8, 16] -> [16, 8, 4]
        call = block.transpose(tile_var, 0, 2)

        assert isinstance(call, ir.Call)
        assert call.op.name == "block.transpose"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert len(result_type.shape) == 3

    def test_row_max_3d(self):
        """Test row_max on 3D tile."""
        span = ir.Span.unknown()

        # Create a 3D tile [4, 16, 32]
        dim4 = ir.ConstInt(4, DataType.INT32, span)
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        dim32 = ir.ConstInt(32, DataType.INT32, span)
        tile_type = ir.TileType([dim4, dim16, dim32], DataType.FP32)
        tile_var = ir.Var("tile", tile_type, span)
        tmp_tile = ir.Var("tmp_tile", tile_type, span)

        # row_max should reduce the last dimension: [4, 16, 32] -> [4, 16, 1]
        call = block.row_max(tile_var, tmp_tile)

        assert isinstance(call, ir.Call)
        assert call.op.name == "block.row_max"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert len(result_type.shape) == 3

    def test_view_3d(self):
        """Test view operation on 3D tile."""
        span = ir.Span.unknown()

        # Create a 3D tile [4, 16, 32]
        dim4 = ir.ConstInt(4, DataType.INT32, span)
        dim16 = ir.ConstInt(16, DataType.INT32, span)
        dim32 = ir.ConstInt(32, DataType.INT32, span)
        tile_type = ir.TileType([dim4, dim16, dim32], DataType.FP16)
        tile_var = ir.Var("tile", tile_type, span)

        # Create a view with different shape [2, 8, 16]
        new_shape = [2, 8, 16]
        offset = [0, 0, 0]
        call = block.view(tile_var, new_shape, offset)

        assert isinstance(call, ir.Call)
        assert call.op.name == "block.view"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert len(result_type.shape) == 3
