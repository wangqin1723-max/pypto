# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Unit tests for block-level reduction operators."""

import pypto.language as pl
from pypto import backend
from pypto.backend import BackendType
from pypto.ir.pass_manager import PassManager

class TestBlockReductionOps:
    """Test suite for block-level reduction operators."""

    def test_block_sum_axis0(self):
        """Test block.sum operator - sum along axis 0 (column-wise)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[1, 32], pl.FP32] = pl.sum(tile_a, axis=0)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [1, 32], output)
                return result

        ir_str = str(Program)
        assert "block.sum" in ir_str

    def test_block_sum_axis1(self):
        """Test block.sum operator - sum along axis 1 (row-wise)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 1], pl.FP32] = pl.sum(tile_a, axis=1)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 1], output)
                return result

        ir_str = str(Program)
        assert "block.sum" in ir_str

    def test_block_max_axis0(self):
        """Test block.max operator - max along axis 0 (column-wise)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[1, 32], pl.FP32] = pl.max(tile_a, axis=0)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [1, 32], output)
                return result

        ir_str = str(Program)
        assert "block.max" in ir_str

    def test_block_max_axis1(self):
        """Test block.max operator - max along axis 1 (row-wise)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 1], pl.FP32] = pl.max(tile_a, axis=1)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 1], output)
                return result

        ir_str = str(Program)
        assert "block.max" in ir_str

    def test_block_row_max(self):
        """Test block.row_max operation."""

        @pl.program
        class RowMaxKernel:
            @pl.function(type=pl.FunctionType.InCore)
            def row_max_kernel(
                self, input: pl.Tensor[[128, 128], pl.FP32], output: pl.Tensor[[128, 1], pl.FP32]
            ) -> pl.Tensor[[128, 1], pl.FP32]:
                tile_in: pl.Tile[[32, 128], pl.FP32] = pl.load(input, [0, 0], [32, 128])
                tmp_tile: pl.Tile[[32, 1], pl.FP32] = pl.block.create_tile(
                    [32, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.UB
                )
                tile_max: pl.Tile[[32, 1], pl.FP32] = pl.row_max(tile_in, tmp_tile)
                result: pl.Tensor[[128, 1], pl.FP32] = pl.store(tile_max, [0, 0], [32, 1], output)
                return result

        program = RowMaxKernel
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.CCE)
        pm = PassManager.get_strategy()
        optimized_program = pm.run_passes(program)

        assert optimized_program is not None
        assert "block.row_max" in str(optimized_program)

    def test_block_row_sum(self):
        """Test block.row_sum operation."""

        @pl.program
        class RowSumKernel:
            @pl.function(type=pl.FunctionType.InCore)
            def row_sum_kernel(
                self, input: pl.Tensor[[128, 128], pl.FP32], output: pl.Tensor[[128, 1], pl.FP32]
            ) -> pl.Tensor[[128, 1], pl.FP32]:
                tile_in: pl.Tile[[32, 128], pl.FP32] = pl.load(input, [0, 0], [32, 128])
                tmp_tile: pl.Tile[[32, 1], pl.FP32] = pl.block.create_tile(
                    [32, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.UB
                )
                tile_sum: pl.Tile[[32, 1], pl.FP32] = pl.row_sum(tile_in, tmp_tile)
                result: pl.Tensor[[128, 1], pl.FP32] = pl.store(tile_sum, [0, 0], [32, 1], output)
                return result

        program = RowSumKernel
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.CCE)
        pm = PassManager.get_strategy()
        optimized_program = pm.run_passes(program)

        assert optimized_program is not None
        assert "block.row_sum" in str(optimized_program)

    def test_block_row_min(self):
        """Test block.row_min operation."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                input: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 1], pl.FP32],
            ) -> pl.Tensor[[128, 1], pl.FP32]:
                tile_in: pl.Tile[[32, 128], pl.FP32] = pl.load(input, [0, 0], [32, 128])
                tmp_tile: pl.Tile[[32, 128], pl.FP32] = pl.block.create_tile(
                    [32, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.UB
                )
                tile_row_min: pl.Tile[[32, 1], pl.FP32] = pl.row_min(tile_in, tmp_tile)
                result: pl.Tensor[[128, 1], pl.FP32] = pl.store(tile_row_min, [0, 0], [32, 1], output)
                return result

        ir_str = str(Program)
        assert "block.row_min" in ir_str

    def test_block_min_axis0(self):
        """Test block.min operator - min along axis 0 (column-wise)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[1, 32], pl.FP32] = pl.min(tile_a, axis=0)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [1, 32], output)
                return result

        ir_str = str(Program)
        assert "block.min" in ir_str

    def test_block_min_axis1(self):
        """Test block.min operator - min along axis 1 (row-wise)."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                output: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                tile_c: pl.Tile[[32, 1], pl.FP32] = pl.min(tile_a, axis=1)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.store(tile_c, [0, 0], [32, 1], output)
                return result

        ir_str = str(Program)
        assert "block.min" in ir_str
