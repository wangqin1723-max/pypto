# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Unit tests for block-level broadcast operators."""

import pypto.language as pl

class TestBlockBroadcastOps:
    """Test suite for block-level broadcast operators."""

    def test_block_col_expand(self):
        """Test block.col_expand operator - expand column vector to target shape."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                target: pl.Tensor[[128, 128], pl.FP32],
                col: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_target: pl.Tile[[32, 32], pl.FP32] = pl.load(target, [0, 0], [32, 32])
                tile_col: pl.Tile[[1, 32], pl.FP32] = pl.load(col, [0, 0], [1, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.col_expand(tile_target, tile_col)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.col_expand" in ir_str

    def test_block_col_expand_mul(self):
        """Test block.col_expand_mul operator - expand column and multiply with tile."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                col: pl.Tensor[[128, 128], pl.FP32],
                tile: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_col: pl.Tile[[1, 32], pl.FP32] = pl.load(col, [0, 0], [1, 32])
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(tile, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.col_expand_mul(tile_a, tile_col)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.col_expand_mul" in ir_str

    def test_block_col_expand_div(self):
        """Test block.col_expand_div operator - expand column and divide tile."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                col: pl.Tensor[[128, 128], pl.FP32],
                tile: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_col: pl.Tile[[1, 32], pl.FP32] = pl.load(col, [0, 0], [1, 32])
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(tile, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.col_expand_div(tile_a, tile_col)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.col_expand_div" in ir_str

    def test_block_col_expand_sub(self):
        """Test block.col_expand_sub operator - expand column and subtract from tile."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                col: pl.Tensor[[128, 128], pl.FP32],
                tile: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_col: pl.Tile[[1, 32], pl.FP32] = pl.load(col, [0, 0], [1, 32])
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(tile, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.col_expand_sub(tile_a, tile_col)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.col_expand_sub" in ir_str

    def test_block_row_expand_add(self):
        """Test block.row_expand_add operator - expand row and add to tile."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                tile: pl.Tensor[[128, 128], pl.FP32],
                row: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(tile, [0, 0], [32, 32])
                tile_row: pl.Tile[[32, 1], pl.FP32] = pl.load(row, [0, 0], [32, 1])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.row_expand_add(tile_a, tile_row)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.row_expand_add" in ir_str

    def test_block_row_expand_sub(self):
        """Test block.row_expand_sub operator - subtract row vector from each tile row."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                tile: pl.Tensor[[128, 128], pl.FP32],
                row: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(tile, [0, 0], [32, 32])
                tile_row: pl.Tile[[32, 1], pl.FP32] = pl.load(row, [0, 0], [32, 1])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.row_expand_sub(tile_a, tile_row)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.row_expand_sub" in ir_str

    def test_block_row_expand_div(self):
        """Test block.row_expand_div operator - divide each tile row by row vector."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                tile: pl.Tensor[[128, 128], pl.FP32],
                row: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(tile, [0, 0], [32, 32])
                tile_row: pl.Tile[[32, 1], pl.FP32] = pl.load(row, [0, 0], [32, 1])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.row_expand_div(tile_a, tile_row)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.row_expand_div" in ir_str

    def test_block_row_expand_mul(self):
        """Test block.row_expand_mul operator - multiply each tile row by row vector."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                tile: pl.Tensor[[128, 128], pl.FP32],
                row: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(tile, [0, 0], [32, 32])
                tile_row: pl.Tile[[32, 1], pl.FP32] = pl.load(row, [0, 0], [32, 1])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.row_expand_mul(tile_a, tile_row)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.row_expand_mul" in ir_str

    def test_block_row_expand(self):
        """Test block.row_expand operator - broadcast first element of each row across the row."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                tile: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(tile, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.row_expand(tile_a)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.row_expand" in ir_str

    def test_block_expands(self):
        """Test block.expands operator - expand scalar to tile shape."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.expands(tile_a, 1.0)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.expands" in ir_str
