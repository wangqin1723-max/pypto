# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Runtime tests for tile-based memory operations using the PyPTO frontend.

Covers all memory ops registered in src/backend/910B_CCE/backend_910b_cce_ops.cpp:
- load + store:  Basic GM→UB→GM memory copy
- full:          Create constant-filled tiles
- move:          Move tiles between memory spaces (L1→UB)
- ub_copy:       Copy tiles within UB memory
- l0c_store:     Store L0C tile to GM (via matmul pipeline)

Each operation has both 32x128 and 128x128 test cases.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

# =============================================================================
# Load + Store: Basic memory copy
# =============================================================================


class TestTileLoadStore32x128(PTOTestCase):
    """Test case for tile load + store (memory copy, 32x128)."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_load_store_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileLoadStoreProgram:
            @pl.function
            def tile_load_store(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                c: pl.Tensor[[32, 128], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 128])
                out_c = pl.store(tile_a, offsets=[0, 0], shapes=[32, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[32, 128], pl.FP32]) -> pl.Tensor[[32, 128], pl.FP32]:
                out_c = self.tile_load_store(a)
                return out_c

        return TileLoadStoreProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"]


class TestTileLoadStore128x128(PTOTestCase):
    """Test case for tile load + store (memory copy, 128x128)."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_load_store_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileLoadStoreProgram:
            @pl.function
            def tile_load_store(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[128, 128])
                out_c = pl.store(tile_a, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_load_store(a)
                return out_c

        return TileLoadStoreProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"]


# =============================================================================
# Full: Create constant-filled tiles
# =============================================================================


class TestTileFull32x128(PTOTestCase):
    """Test case for tile full (constant fill, 32x128)."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_full_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("c", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileFullProgram:
            @pl.function
            def tile_full(
                self,
                c: pl.Tensor[[32, 128], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                tile_c = pl.block.full([32, 128], pl.FP32, 3.0)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[32, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self) -> pl.Tensor[[32, 128], pl.FP32]:
                out_c = self.tile_full()
                return out_c

        return TileFullProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = 3.0


class TestTileFull128x128(PTOTestCase):
    """Test case for tile full (constant fill, 128x128)."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_full_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileFullProgram:
            @pl.function
            def tile_full(
                self,
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_c = pl.block.full([128, 128], pl.FP32, 3.0)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_full()
                return out_c

        return TileFullProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = 3.0


# =============================================================================
# Move: Move tiles between memory spaces (L1 → UB)
# =============================================================================


class TestTileMove32x128(PTOTestCase):
    """Test case for tile move L1→UB (32x128)."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_move_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileMoveProgram:
            @pl.function
            def tile_move(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                c: pl.Tensor[[32, 128], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                tile_a_l1 = pl.load(a, offsets=[0, 0], shapes=[32, 128], target_memory=pl.MemorySpace.L1)
                tile_a_ub = pl.move(tile_a_l1, target_memory=pl.MemorySpace.UB)
                out_c = pl.store(tile_a_ub, offsets=[0, 0], shapes=[32, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[32, 128], pl.FP32]) -> pl.Tensor[[32, 128], pl.FP32]:
                out_c = self.tile_move(a)
                return out_c

        return TileMoveProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"]


class TestTileMove128x128(PTOTestCase):
    """Test case for tile move L1→UB (128x128)."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_move_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileMoveProgram:
            @pl.function
            def tile_move(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a_l1 = pl.load(a, offsets=[0, 0], shapes=[128, 128], target_memory=pl.MemorySpace.L1)
                tile_a_ub = pl.move(tile_a_l1, target_memory=pl.MemorySpace.UB)
                out_c = pl.store(tile_a_ub, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_move(a)
                return out_c

        return TileMoveProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"]


# =============================================================================
# UbCopy: Copy tiles within UB memory
# =============================================================================


class TestTileUbCopy32x128(PTOTestCase):
    """Test case for tile ub_copy (UB→UB copy, 32x128)."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_ub_copy_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileUbCopyProgram:
            @pl.function
            def tile_ub_copy(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                c: pl.Tensor[[32, 128], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 128])
                tile_copy = pl.ub_copy(tile_a)
                out_c = pl.store(tile_copy, offsets=[0, 0], shapes=[32, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[32, 128], pl.FP32]) -> pl.Tensor[[32, 128], pl.FP32]:
                out_c = self.tile_ub_copy(a)
                return out_c

        return TileUbCopyProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"]


class TestTileUbCopy128x128(PTOTestCase):
    """Test case for tile ub_copy (UB→UB copy, 128x128)."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_ub_copy_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileUbCopyProgram:
            @pl.function
            def tile_ub_copy(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[128, 128])
                tile_copy = pl.ub_copy(tile_a)
                out_c = pl.store(tile_copy, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_ub_copy(a)
                return out_c

        return TileUbCopyProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"]


# =============================================================================
# L0CStore: Store L0C tile to GM (via matmul pipeline)
# =============================================================================


class TestTileL0CStore32x128(PTOTestCase):
    """Test case for l0c_store: load→move→matmul→l0c_store (A[32,128] @ B[128,128] → C[32,128])."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_l0c_store_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 128], DataType.FP32, init_value=2.0),
            TensorSpec("b", [128, 128], DataType.FP32, init_value=3.0),
            TensorSpec("c", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileL0CStoreProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_l0c_store(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[32, 128], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                tile_a_l1 = pl.load(a, offsets=[0, 0], shapes=[32, 128], target_memory=pl.MemorySpace.L1)
                tile_b_l1 = pl.load(b, offsets=[0, 0], shapes=[128, 128], target_memory=pl.MemorySpace.L1)
                tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.L0A)
                tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.L0B)
                tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
                out_c = pl.l0c_store(tile_c_l0c, offsets=[0, 0], shapes=[32, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[32, 128], pl.FP32], b: pl.Tensor[[128, 128], pl.FP32]
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                out_c = self.tile_l0c_store(a, b)
                return out_c

        return TileL0CStoreProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.matmul(tensors["a"], tensors["b"])


class TestTileL0CStore128x128(PTOTestCase):
    """Test case for l0c_store: load→move→matmul→l0c_store (A[128,128] @ B[128,128] → C[128,128])."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_l0c_store_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=2.0),
            TensorSpec("b", [128, 128], DataType.FP32, init_value=3.0),
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileL0CStoreProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_l0c_store(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a_l1 = pl.load(a, offsets=[0, 0], shapes=[128, 128], target_memory=pl.MemorySpace.L1)
                tile_b_l1 = pl.load(b, offsets=[0, 0], shapes=[128, 128], target_memory=pl.MemorySpace.L1)
                tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.L0A)
                tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.L0B)
                tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
                out_c = pl.l0c_store(tile_c_l0c, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[128, 128], pl.FP32], b: pl.Tensor[[128, 128], pl.FP32]
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_l0c_store(a, b)
                return out_c

        return TileL0CStoreProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.matmul(tensors["a"], tensors["b"])


# =============================================================================
# pytest test functions
# =============================================================================


class TestMemoryOperations:
    """Test suite for memory operations."""

    def test_tile_load_store_128x128(self, test_runner):
        """Test tile load + store with 128x128 shape."""
        test_case = TestTileLoadStore128x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"

    def test_tile_load_store_32x128(self, test_runner):
        """Test tile load + store with 32x128 shape."""
        test_case = TestTileLoadStore32x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 32x128: {result.error}"

    def test_tile_full_32x128(self, test_runner):
        """Test tile full (constant fill) with 32x128 shape."""
        test_case = TestTileFull32x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 32x128: {result.error}"

    def test_tile_full_128x128(self, test_runner):
        """Test tile full (constant fill) with 128x128 shape."""
        test_case = TestTileFull128x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"

    def test_tile_move_32x128(self, test_runner):
        """Test tile move L1→UB with 32x128 shape."""
        test_case = TestTileMove32x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 32x128: {result.error}"

    def test_tile_move_128x128(self, test_runner):
        """Test tile move L1→UB with 128x128 shape."""
        test_case = TestTileMove128x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"

    def test_tile_ub_copy_32x128(self, test_runner):
        """Test tile ub_copy (UB→UB) with 32x128 shape."""
        test_case = TestTileUbCopy32x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 32x128: {result.error}"

    def test_tile_ub_copy_128x128(self, test_runner):
        """Test tile ub_copy (UB→UB) with 128x128 shape."""
        test_case = TestTileUbCopy128x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"

    def test_tile_l0c_store_32x128(self, test_runner):
        """Test l0c_store via matmul pipeline with A[32,128] @ B[128,128] → C[32,128]."""
        test_case = TestTileL0CStore32x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 32x128: {result.error}"

    def test_tile_l0c_store_128x128(self, test_runner):
        """Test l0c_store via matmul pipeline with A[128,128] @ B[128,128] → C[128,128]."""
        test_case = TestTileL0CStore128x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
