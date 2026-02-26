# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Unit tests for block-level unary operators."""

import pypto.language as pl

class TestBlockUnaryOps:
    """Test suite for block-level unary operators."""

    def test_block_log(self):
        """Test block.log operator - natural logarithm of all elements."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.log(tile_a)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.log" in ir_str

    def test_block_abs(self):
        """Test block.abs operator - absolute value of all elements."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.abs(tile_a)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.abs" in ir_str

    def test_block_relu(self):
        """Test block.relu operator - ReLU activation function."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.relu(tile_a)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.relu" in ir_str

    def test_block_exp(self):
        """Test block.exp operator - exponential of all elements."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.exp(tile_a)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.exp" in ir_str

    def test_block_sqrt(self):
        """Test block.sqrt operator - square root of all elements."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.sqrt(tile_a)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.sqrt" in ir_str

    def test_block_neg(self):
        """Test block.neg operator - negate all elements."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.neg(tile_a)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.neg" in ir_str
