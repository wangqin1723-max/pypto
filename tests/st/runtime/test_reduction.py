# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Runtime tests for tile-based reduction operations using the PyPTO frontend.

Covers:
  - block.sum, block.max, block.min — axis-based reductions (CCE-only)
    - axis=0: column-wise (col_sum / col_max / col_min)
    - axis=1: row-wise (row_sum / row_max / row_min)
  - block.row_max, block.row_min — row-wise reductions (explicit tmp tile)
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec


# =============================================================================
# block.sum (axis=1, row-wise)
# =============================================================================


class TestTileRowSum(PTOTestCase):
    """Test block.sum axis=1: [64, 64] -> [64, 1]."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_row_sum_64x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [64, 64], DataType.FP32, init_value=1.0),
            TensorSpec("c", [64, 1], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileRowSumProgram:
            @pl.function
            def tile_row_sum(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                c: pl.Tensor[[64, 1], pl.FP32],
            ) -> pl.Tensor[[64, 1], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[64, 64])
                tile_c = pl.sum(tile_a, axis=1, keepdim=True)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[64, 1], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[64, 64], pl.FP32]
            ) -> pl.Tensor[[64, 1], pl.FP32]:
                out_c: pl.Tensor[[64, 1], pl.FP32] = self.tile_row_sum(a)
                return out_c

        return TileRowSumProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.sum(tensors["a"], dim=1, keepdim=True)


class TestTileRowSum16x64(PTOTestCase):
    """Test block.sum axis=1: [16, 64] -> [16, 1]."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_row_sum_16x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [16, 64], DataType.FP32, init_value=1.0),
            TensorSpec("c", [16, 1], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileRowSumProgram:
            @pl.function
            def tile_row_sum(
                self,
                a: pl.Tensor[[16, 64], pl.FP32],
                c: pl.Tensor[[16, 1], pl.FP32],
            ) -> pl.Tensor[[16, 1], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[16, 64])
                tile_c = pl.sum(tile_a, axis=1, keepdim=True)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[16, 1], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[16, 64], pl.FP32]
            ) -> pl.Tensor[[16, 1], pl.FP32]:
                out_c: pl.Tensor[[16, 1], pl.FP32] = self.tile_row_sum(a)
                return out_c

        return TileRowSumProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.sum(tensors["a"], dim=1, keepdim=True)


# =============================================================================
# block.row_max
# =============================================================================


class TestTileRowMax(PTOTestCase):
    """Test block.row_max: [64, 64] -> [64, 1]."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_row_max_64x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [64, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [64, 1], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileRowMaxProgram:
            @pl.function
            def tile_row_max(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Tensor[[64, 1], pl.FP32],
            ) -> pl.Tensor[[64, 1], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(a, offsets=[0, 0], shapes=[64, 64])
                tmp: pl.Tile[[64, 64], pl.FP32] = pl.create_tile([64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.UB)
                tile_c: pl.Tile[[64, 1], pl.FP32] = pl.row_max(tile_a, tmp)
                out_c: pl.Tile[[64, 1], pl.FP32] = pl.store(tile_c, offsets=[0, 0], shapes=[64, 1], output_tensor=output)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[64, 64], pl.FP32]
            ) -> pl.Tensor[[64, 1], pl.FP32]:
                out_c: pl.Tensor[[64, 1], pl.FP32] = self.tile_row_max(a)
                return out_c

        return TileRowMaxProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.max(tensors["a"], dim=1, keepdim=True).values


class TestTileRowMax16x64(PTOTestCase):
    """Test block.row_max: [16, 64] -> [16, 1]."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_row_max_16x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [16, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [16, 1], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileRowMaxProgram:
            @pl.function
            def tile_row_max(
                self,
                a: pl.Tensor[[16, 64], pl.FP32],
                c: pl.Tensor[[16, 1], pl.FP32],
            ) -> pl.Tensor[[16, 1], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[16, 64])
                tmp = pl.create_tile([16, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.UB)
                tile_c = pl.row_max(tile_a, tmp)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[16, 1], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[16, 64], pl.FP32]
            ) -> pl.Tensor[[16, 1], pl.FP32]:
                out_c: pl.Tensor[[16, 1], pl.FP32] = self.tile_row_max(a)
                return out_c

        return TileRowMaxProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.max(tensors["a"], dim=1, keepdim=True).values


# =============================================================================
# block.row_min
# =============================================================================


class TestTileRowMin(PTOTestCase):
    """Test block.row_min: [64, 64] -> [64, 1]."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_row_min_64x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [64, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [64, 1], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileRowMinProgram:
            @pl.function
            def tile_row_min(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                c: pl.Tensor[[64, 1], pl.FP32],
            ) -> pl.Tensor[[64, 1], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[64, 64])
                tmp = pl.create_tile([64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.UB)
                tile_c = pl.row_min(tile_a, tmp)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[64, 1], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[64, 64], pl.FP32]
            ) -> pl.Tensor[[64, 1], pl.FP32]:
                out_c: pl.Tensor[[64, 1], pl.FP32] = self.tile_row_min(a)
                return out_c

        return TileRowMinProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.min(tensors["a"], dim=1, keepdim=True).values


class TestTileRowMin16x64(PTOTestCase):
    """Test block.row_min: [16, 64] -> [16, 1]."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_row_min_16x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [16, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [16, 1], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileRowMinProgram:
            @pl.function
            def tile_row_min(
                self,
                a: pl.Tensor[[16, 64], pl.FP32],
                c: pl.Tensor[[16, 1], pl.FP32],
            ) -> pl.Tensor[[16, 1], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[16, 64])
                tmp = pl.create_tile([16, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.UB)
                tile_c = pl.row_min(tile_a, tmp)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[16, 1], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[16, 64], pl.FP32]
            ) -> pl.Tensor[[16, 1], pl.FP32]:
                out_c: pl.Tensor[[16, 1], pl.FP32] = self.tile_row_min(a)
                return out_c

        return TileRowMinProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.min(tensors["a"], dim=1, keepdim=True).values


# =============================================================================
# pytest test suite
# =============================================================================


class TestRowSumOperations:
    """Test suite for block.sum axis=1 (row-wise)."""

    def test_tile_row_sum_64x64(self, test_runner):
        """Test block.row_sum (64x64)."""
        result = test_runner.run(TestTileRowSum())
        assert result.passed, f"Test failed: {result.error}"

    def test_tile_row_sum_16x64(self, test_runner):
        """Test block.row_sum (16x64)."""
        result = test_runner.run(TestTileRowSum16x64())
        assert result.passed, f"Test failed: {result.error}"


class TestRowMaxOperations:
    """Test suite for block.row_max."""

    def test_tile_row_max_64x64(self, test_runner):
        """Test block.row_max (64x64)."""
        result = test_runner.run(TestTileRowMax())
        assert result.passed, f"Test failed: {result.error}"

    def test_tile_row_max_16x64(self, test_runner):
        """Test block.row_max (16x64)."""
        result = test_runner.run(TestTileRowMax16x64())
        assert result.passed, f"Test failed: {result.error}"


class TestRowMinOperations:
    """Test suite for block.row_min."""

    def test_tile_row_min_64x64(self, test_runner):
        """Test block.row_min (64x64)."""
        result = test_runner.run(TestTileRowMin())
        assert result.passed, f"Test failed: {result.error}"

    def test_tile_row_min_16x64(self, test_runner):
        """Test block.row_min (16x64)."""
        result = test_runner.run(TestTileRowMin16x64())
        assert result.passed, f"Test failed: {result.error}"


# =============================================================================
# block.sum (axis=0, col-wise) — CCE-only
# =============================================================================


class TestTileColSum(PTOTestCase):
    """Test block.sum axis=0: [64, 64] -> [1, 64]."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_col_sum_64x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [64, 64], DataType.FP32, init_value=1.0),
            TensorSpec("c", [1, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileColSumProgram:
            @pl.function
            def tile_col_sum(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                c: pl.Tensor[[1, 64], pl.FP32],
            ) -> pl.Tensor[[1, 64], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[64, 64])
                tile_c = pl.sum(tile_a, axis=0, keepdim=True)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[1, 64], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[64, 64], pl.FP32]
            ) -> pl.Tensor[[1, 64], pl.FP32]:
                out_c: pl.Tensor[[1, 64], pl.FP32] = self.tile_col_sum(a)
                return out_c

        return TileColSumProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.sum(tensors["a"], dim=0, keepdim=True)


class TestTileColSum16x64(PTOTestCase):
    """Test block.sum axis=0: [16, 64] -> [1, 64]."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_col_sum_16x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [16, 64], DataType.FP32, init_value=1.0),
            TensorSpec("c", [1, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileColSumProgram:
            @pl.function
            def tile_col_sum(
                self,
                a: pl.Tensor[[16, 64], pl.FP32],
                c: pl.Tensor[[1, 64], pl.FP32],
            ) -> pl.Tensor[[1, 64], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[16, 64])
                tile_c = pl.sum(tile_a, axis=0, keepdim=True)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[1, 64], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[16, 64], pl.FP32]
            ) -> pl.Tensor[[1, 64], pl.FP32]:
                out_c: pl.Tensor[[1, 64], pl.FP32] = self.tile_col_sum(a)
                return out_c

        return TileColSumProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.sum(tensors["a"], dim=0, keepdim=True)


# =============================================================================
# block.max (axis=0, col-wise) — CCE-only
# =============================================================================


class TestTileColMax(PTOTestCase):
    """Test block.max axis=0: [64, 64] -> [1, 64]."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_col_max_64x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [64, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [1, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileColMaxProgram:
            @pl.function
            def tile_col_max(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                c: pl.Tensor[[1, 64], pl.FP32],
            ) -> pl.Tensor[[1, 64], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[64, 64])
                tile_c = pl.max(tile_a, axis=0, keepdim=True)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[1, 64], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[64, 64], pl.FP32]
            ) -> pl.Tensor[[1, 64], pl.FP32]:
                out_c: pl.Tensor[[1, 64], pl.FP32] = self.tile_col_max(a)
                return out_c

        return TileColMaxProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.max(tensors["a"], dim=0, keepdim=True).values


class TestTileColMax16x64(PTOTestCase):
    """Test block.max axis=0: [16, 64] -> [1, 64]."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_col_max_16x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [16, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [1, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileColMaxProgram:
            @pl.function
            def tile_col_max(
                self,
                a: pl.Tensor[[16, 64], pl.FP32],
                c: pl.Tensor[[1, 64], pl.FP32],
            ) -> pl.Tensor[[1, 64], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[16, 64])
                tile_c = pl.max(tile_a, axis=0, keepdim=True)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[1, 64], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[16, 64], pl.FP32]
            ) -> pl.Tensor[[1, 64], pl.FP32]:
                out_c: pl.Tensor[[1, 64], pl.FP32] = self.tile_col_max(a)
                return out_c

        return TileColMaxProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.max(tensors["a"], dim=0, keepdim=True).values


# =============================================================================
# block.min (axis=0, col-wise) — CCE-only
# =============================================================================


class TestTileColMin(PTOTestCase):
    """Test block.min axis=0: [64, 64] -> [1, 64]."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_col_min_64x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [64, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [1, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileColMinProgram:
            @pl.function
            def tile_col_min(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                c: pl.Tensor[[1, 64], pl.FP32],
            ) -> pl.Tensor[[1, 64], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[64, 64])
                tile_c = pl.min(tile_a, axis=0, keepdim=True)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[1, 64], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[64, 64], pl.FP32]
            ) -> pl.Tensor[[1, 64], pl.FP32]:
                out_c: pl.Tensor[[1, 64], pl.FP32] = self.tile_col_min(a)
                return out_c

        return TileColMinProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.min(tensors["a"], dim=0, keepdim=True).values


class TestTileColMin16x64(PTOTestCase):
    """Test block.min axis=0: [16, 64] -> [1, 64]."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_col_min_16x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [16, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [1, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileColMinProgram:
            @pl.function
            def tile_col_min(
                self,
                a: pl.Tensor[[16, 64], pl.FP32],
                c: pl.Tensor[[1, 64], pl.FP32],
            ) -> pl.Tensor[[1, 64], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[16, 64])
                tile_c = pl.min(tile_a, axis=0, keepdim=True)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[1, 64], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[16, 64], pl.FP32]
            ) -> pl.Tensor[[1, 64], pl.FP32]:
                out_c: pl.Tensor[[1, 64], pl.FP32] = self.tile_col_min(a)
                return out_c

        return TileColMinProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.min(tensors["a"], dim=0, keepdim=True).values


# =============================================================================
# pytest test suite (axis-based col-wise, CCE-only)
# =============================================================================


class TestColSumOperations:
    """Test suite for block.sum axis=0."""

    def test_tile_col_sum_64x64(self, test_runner):
        """Test block.sum axis=0 (64x64)."""
        result = test_runner.run(TestTileColSum())
        assert result.passed, f"Test failed: {result.error}"

    def test_tile_col_sum_16x64(self, test_runner):
        """Test block.sum axis=0 (16x64)."""
        result = test_runner.run(TestTileColSum16x64())
        assert result.passed, f"Test failed: {result.error}"


class TestColMaxOperations:
    """Test suite for block.max axis=0."""

    def test_tile_col_max_64x64(self, test_runner):
        """Test block.max axis=0 (64x64)."""
        result = test_runner.run(TestTileColMax())
        assert result.passed, f"Test failed: {result.error}"

    def test_tile_col_max_16x64(self, test_runner):
        """Test block.max axis=0 (16x64)."""
        result = test_runner.run(TestTileColMax16x64())
        assert result.passed, f"Test failed: {result.error}"


class TestColMinOperations:
    """Test suite for block.min axis=0."""

    def test_tile_col_min_64x64(self, test_runner):
        """Test block.min axis=0 (64x64)."""
        result = test_runner.run(TestTileColMin())
        assert result.passed, f"Test failed: {result.error}"

    def test_tile_col_min_16x64(self, test_runner):
        """Test block.min axis=0 (16x64)."""
        result = test_runner.run(TestTileColMin16x64())
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
