# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Unit tests for block-level element-wise and bitwise/arithmetic operators."""

import pypto.language as pl


class TestBlockElementwiseOps:
    """Test suite for block-level element-wise operators (tile-tile and tile-scalar)."""

    def test_block_add(self):
        """Test block.add operator - element-wise addition of two tiles."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32] = pl.load(b, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.add(tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.add" in ir_str

    def test_block_sub(self):
        """Test block.sub operator - element-wise subtraction of two tiles."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32] = pl.load(b, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.sub(tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.sub" in ir_str

    def test_block_mul(self):
        """Test block.mul operator - element-wise multiplication of two tiles."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32] = pl.load(b, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.mul(tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.mul" in ir_str

    def test_block_div(self):
        """Test block.div operator - element-wise division of two tiles."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32] = pl.load(b, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.div(tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.div" in ir_str

    def test_block_muls(self):
        """Test block.muls operator - multiply all elements of a tile by scalar."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.mul(tile_a, 2.0)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.muls" in ir_str

    def test_block_cmp(self):
        """Test block.cmp operator - element-wise comparison of two tiles."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32] = pl.load(b, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.cmp(tile_a, tile_b, cmp_type=0)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.cmp" in ir_str

    def test_block_cmps(self):
        """Test block.cmps operator - compare tile elements with scalar."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.cmps(tile_a, 0.0, cmp_type=0)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.cmps" in ir_str


class TestBlockBitwiseArithmeticOps:
    """Test suite for newly added block-level bitwise and arithmetic ops (rem, and, or, xor)."""

    def test_block_rem(self):
        """Test block.rem operator - element-wise remainder of two tiles."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32] = pl.load(b, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.rem(tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.rem" in ir_str

    def test_block_rems(self):
        """Test block.rems operator - element-wise remainder of tile and scalar."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.rems(tile_a, 3.0)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.rems" in ir_str

    def test_block_and(self):
        """Test block.and operator - element-wise bitwise AND of two tiles."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.INT32],
                b: pl.Tensor[[128, 128], pl.INT32],
                output: pl.Tensor[[128, 128], pl.INT32],
            ) -> pl.Tensor[[128, 128], pl.INT32]:
                tile_a: pl.Tile[[32, 32], pl.INT32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.INT32] = pl.load(b, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.INT32] = pl.and_(tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.INT32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.and" in ir_str

    def test_block_ands(self):
        """Test block.ands operator - element-wise bitwise AND of tile and scalar."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.INT32],
                scalar: pl.Scalar[pl.INT32],
                output: pl.Tensor[[128, 128], pl.INT32],
            ) -> pl.Tensor[[128, 128], pl.INT32]:
                tile_a: pl.Tile[[32, 32], pl.INT32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.INT32] = pl.ands(tile_a, scalar)
                result: pl.Tensor[[128, 128], pl.INT32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.ands" in ir_str

    def test_block_or(self):
        """Test block.or operator - element-wise bitwise OR of two tiles."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.INT32],
                b: pl.Tensor[[128, 128], pl.INT32],
                output: pl.Tensor[[128, 128], pl.INT32],
            ) -> pl.Tensor[[128, 128], pl.INT32]:
                tile_a: pl.Tile[[32, 32], pl.INT32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.INT32] = pl.load(b, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.INT32] = pl.or_(tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.INT32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.or" in ir_str

    def test_block_ors(self):
        """Test block.ors operator - element-wise bitwise OR of tile and scalar."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.INT32],
                scalar: pl.Scalar[pl.INT32],
                output: pl.Tensor[[128, 128], pl.INT32],
            ) -> pl.Tensor[[128, 128], pl.INT32]:
                tile_a: pl.Tile[[32, 32], pl.INT32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.INT32] = pl.ors(tile_a, scalar)
                result: pl.Tensor[[128, 128], pl.INT32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.ors" in ir_str

    def test_block_xor(self):
        """Test block.xor operator - element-wise bitwise XOR of two tiles with tmp buffer."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.INT32],
                b: pl.Tensor[[128, 128], pl.INT32],
                output: pl.Tensor[[128, 128], pl.INT32],
            ) -> pl.Tensor[[128, 128], pl.INT32]:
                tile_a: pl.Tile[[32, 32], pl.INT32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.INT32] = pl.load(b, [0, 0], [32, 32])
                tmp: pl.Tile[[32, 32], pl.INT32] = pl.block.create_tile(
                    [32, 32], dtype=pl.INT32, target_memory=pl.MemorySpace.UB
                )
                tile_c: pl.Tile[[32, 32], pl.INT32] = pl.xor(tile_a, tile_b, tmp)
                result: pl.Tensor[[128, 128], pl.INT32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.xor" in ir_str

    def test_block_xors(self):
        """Test block.xors operator - element-wise bitwise XOR of tile and scalar with tmp buffer."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.INT32],
                scalar: pl.Scalar[pl.INT32],
                output: pl.Tensor[[128, 128], pl.INT32],
            ) -> pl.Tensor[[128, 128], pl.INT32]:
                tile_a: pl.Tile[[32, 32], pl.INT32] = pl.load(a, [0, 0], [32, 32])
                tmp: pl.Tile[[32, 32], pl.INT32] = pl.block.create_tile(
                    [32, 32], dtype=pl.INT32, target_memory=pl.MemorySpace.UB
                )
                tile_c: pl.Tile[[32, 32], pl.INT32] = pl.xors(tile_a, scalar, tmp)
                result: pl.Tensor[[128, 128], pl.INT32] = pl.store(tile_c, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.xors" in ir_str

    def test_block_shl(self):
        """Test block.shl operator - element-wise bitwise left shift of two tiles."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.UINT32],
                b: pl.Tensor[[128, 128], pl.UINT32],
                output: pl.Tensor[[128, 128], pl.UINT32],
            ) -> pl.Tensor[[128, 128], pl.UINT32]:
                tile_a: pl.Tile[[16, 16], pl.UINT32] = pl.load(a, [0, 0], [16, 16])
                tile_b: pl.Tile[[16, 16], pl.UINT32] = pl.load(b, [0, 0], [16, 16])
                tile_c: pl.Tile[[16, 16], pl.UINT32] = pl.shl(tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.UINT32] = pl.store(tile_c, [0, 0], [16, 16], output)
                return result

        ir_str = str(Program)
        assert "block.shl" in ir_str

    def test_block_shls(self):
        """Test block.shls operator - element-wise bitwise left shift of tile and scalar."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.UINT32],
                scalar: pl.Scalar[pl.INT32],
                output: pl.Tensor[[128, 128], pl.UINT32],
            ) -> pl.Tensor[[128, 128], pl.UINT32]:
                tile_a: pl.Tile[[16, 16], pl.UINT32] = pl.load(a, [0, 0], [16, 16])
                tile_c: pl.Tile[[16, 16], pl.UINT32] = pl.shls(tile_a, scalar)
                result: pl.Tensor[[128, 128], pl.UINT32] = pl.store(tile_c, [0, 0], [16, 16], output)
                return result

        ir_str = str(Program)
        assert "block.shls" in ir_str

    def test_block_maxs(self):
        """Test block.maxs operator - element-wise maximum of tile and scalar."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                tile_c: pl.Tile[[16, 16], pl.FP32] = pl.maxs(tile_a, 0.0)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [16, 16], output)
                return result

        ir_str = str(Program)
        assert "block.maxs" in ir_str

    def test_block_mins(self):
        """Test block.mins operator - element-wise minimum of tile and scalar."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                tile_c: pl.Tile[[16, 16], pl.FP32] = pl.mins(tile_a, 0.0)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [16, 16], output)
                return result

        ir_str = str(Program)
        assert "block.mins" in ir_str

    def test_block_shr(self):
        """Test block.shr operator - element-wise bitwise right shift of two tiles."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.UINT32],
                b: pl.Tensor[[128, 128], pl.UINT32],
                output: pl.Tensor[[128, 128], pl.UINT32],
            ) -> pl.Tensor[[128, 128], pl.UINT32]:
                tile_a: pl.Tile[[16, 16], pl.UINT32] = pl.load(a, [0, 0], [16, 16])
                tile_b: pl.Tile[[16, 16], pl.UINT32] = pl.load(b, [0, 0], [16, 16])
                tile_c: pl.Tile[[16, 16], pl.UINT32] = pl.shr(tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.UINT32] = pl.store(tile_c, [0, 0], [16, 16], output)
                return result

        ir_str = str(Program)
        assert "block.shr" in ir_str

    def test_block_shrs(self):
        """Test block.shrs operator - element-wise bitwise right shift of tile and scalar."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.UINT32],
                scalar: pl.Scalar[pl.INT32],
                output: pl.Tensor[[128, 128], pl.UINT32],
            ) -> pl.Tensor[[128, 128], pl.UINT32]:
                tile_a: pl.Tile[[16, 16], pl.UINT32] = pl.load(a, [0, 0], [16, 16])
                tile_c: pl.Tile[[16, 16], pl.UINT32] = pl.shrs(tile_a, scalar)
                result: pl.Tensor[[128, 128], pl.UINT32] = pl.store(tile_c, [0, 0], [16, 16], output)
                return result

        ir_str = str(Program)
        assert "block.shrs" in ir_str

    def test_block_shl_preserves_lhs_dtype(self):
        """Regression: block.shl result dtype must match LHS dtype, not the promoted type.

        When lhs is UINT16 and rhs is UINT32, the result must be UINT16 (LHS dtype),
        consistent with the scalar variant block.shls which preserves the LHS tile dtype.
        """

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.UINT16],
                b: pl.Tensor[[128, 128], pl.UINT32],
                output: pl.Tensor[[128, 128], pl.UINT16],
            ) -> pl.Tensor[[128, 128], pl.UINT16]:
                tile_a: pl.Tile[[16, 16], pl.UINT16] = pl.load(a, [0, 0], [16, 16])
                tile_b: pl.Tile[[16, 16], pl.UINT32] = pl.load(b, [0, 0], [16, 16])
                tile_c: pl.Tile[[16, 16], pl.UINT16] = pl.shl(tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.UINT16] = pl.store(tile_c, [0, 0], [16, 16], output)
                return result

        ir_str = str(Program)
        assert "block.shl" in ir_str

    def test_block_shr_preserves_lhs_dtype(self):
        """Regression: block.shr result dtype must match LHS dtype, not the promoted type.

        When lhs is UINT16 and rhs is UINT32, the result must be UINT16 (LHS dtype),
        consistent with the scalar variant block.shrs which preserves the LHS tile dtype.
        """

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.UINT16],
                b: pl.Tensor[[128, 128], pl.UINT32],
                output: pl.Tensor[[128, 128], pl.UINT16],
            ) -> pl.Tensor[[128, 128], pl.UINT16]:
                tile_a: pl.Tile[[16, 16], pl.UINT16] = pl.load(a, [0, 0], [16, 16])
                tile_b: pl.Tile[[16, 16], pl.UINT32] = pl.load(b, [0, 0], [16, 16])
                tile_c: pl.Tile[[16, 16], pl.UINT16] = pl.shr(tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.UINT16] = pl.store(tile_c, [0, 0], [16, 16], output)
                return result

        ir_str = str(Program)
        assert "block.shr" in ir_str

    def test_block_prelu(self):
        """Test block.prelu operator - element-wise parametric ReLU with slope and tmp buffer."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_x: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                slope: pl.Tile[[16, 16], pl.FP32] = pl.block.create_tile(
                    [16, 16], dtype=pl.FP32, target_memory=pl.MemorySpace.UB
                )
                tmp: pl.Tile[[16, 16], pl.FP32] = pl.block.create_tile(
                    [16, 16], dtype=pl.FP32, target_memory=pl.MemorySpace.UB
                )
                tile_c: pl.Tile[[16, 16], pl.FP32] = pl.prelu(tile_x, slope, tmp)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [16, 16], output)
                return result

        ir_str = str(Program)
        assert "block.prelu" in ir_str

    def test_block_not(self):
        """Test block.not operator - element-wise bitwise NOT of a tile (int16/uint16 only)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.INT16],
                output: pl.Tensor[[128, 128], pl.INT16],
            ) -> pl.Tensor[[128, 128], pl.INT16]:
                tile_a: pl.Tile[[16, 16], pl.INT16] = pl.load(a, [0, 0], [16, 16])
                tile_c: pl.Tile[[16, 16], pl.INT16] = pl.not_(tile_a)
                result: pl.Tensor[[128, 128], pl.INT16] = pl.store(tile_c, [0, 0], [16, 16], output)
                return result

        ir_str = str(Program)
        assert "block.not" in ir_str

    def test_block_addc(self):
        """Test block.addc operator - element-wise addition of three tiles."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32] = pl.load(b, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.load(c, [0, 0], [32, 32])
                tile_out: pl.Tile[[32, 32], pl.FP32] = pl.addc(tile_a, tile_b, tile_c)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_out, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.addc" in ir_str

    def test_block_subc(self):
        """Test block.subc operator - element-wise subtraction of three tiles."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32] = pl.load(b, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 32], pl.FP32] = pl.load(c, [0, 0], [32, 32])
                tile_out: pl.Tile[[32, 32], pl.FP32] = pl.subc(tile_a, tile_b, tile_c)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_out, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.subc" in ir_str

    def test_block_addsc(self):
        """Test block.addsc operator - element-wise addition of tile, scalar, and tile."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32] = pl.load(b, [0, 0], [32, 32])
                tile_out: pl.Tile[[32, 32], pl.FP32] = pl.addsc(tile_a, 2.0, tile_b)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_out, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.addsc" in ir_str

    def test_block_subsc(self):
        """Test block.subsc operator - element-wise subtraction of tile, scalar, and tile."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32] = pl.load(b, [0, 0], [32, 32])
                tile_out: pl.Tile[[32, 32], pl.FP32] = pl.subsc(tile_a, 2.0, tile_b)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_out, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.subsc" in ir_str

    def test_block_lrelu(self):
        """Test block.lrelu operator - element-wise leaky ReLU with scalar slope."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_out: pl.Tile[[32, 32], pl.FP32] = pl.lrelu(tile_a, 0.1)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_out, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.lrelu" in ir_str

    def test_block_sels(self):
        """Test block.sels operator - select between two tiles via integer scalar mode."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32] = pl.load(b, [0, 0], [32, 32])
                tile_out: pl.Tile[[32, 32], pl.FP32] = pl.sels(tile_a, tile_b, 1)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_out, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.sels" in ir_str

    def test_block_sel(self):
        """Test block.sel operator - per-element selection between two tiles via mask tile."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                m: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_b: pl.Tile[[32, 32], pl.FP32] = pl.load(b, [0, 0], [32, 32])
                tile_m: pl.Tile[[32, 32], pl.FP32] = pl.load(m, [0, 0], [32, 32])
                tile_out: pl.Tile[[32, 32], pl.FP32] = pl.sel(tile_m, tile_a, tile_b)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_out, [0, 0], [32, 32], output)
                return result

        ir_str = str(Program)
        assert "block.sel" in ir_str
