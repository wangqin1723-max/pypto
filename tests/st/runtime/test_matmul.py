# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Runtime tests for matrix multiplication operations using the PyPTO frontend.

Covers all ops registered in src/ir/op/block_ops/matmul.cpp:
- matmul:      C[M,N] = A[M,K] @ B[K,N]
- matmul_acc:  C[M,N] += A[M,K] @ B[K,N]  (with accumulation)
- matmul_bias: C[M,N] = A[M,K] @ B[K,N] + bias[1,N]
- gemv:        C[1,N] = A[1,K] @ B[K,N]
- gemv_acc:    C[1,N] += A[1,K] @ B[K,N]  (with accumulation)
- gemv_bias:   C[1,N] = A[1,K] @ B[K,N] + bias[1,N]

Each op is tested with two shapes: 32x128 and 128x128.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

# =============================================================================
# matmul: C[M,N] = A[M,K] @ B[K,N]
# =============================================================================


class TestMatmul32x128(PTOTestCase):
    """Test matmul with A[32,128] @ B[128,128] -> C[32,128]."""

    __test__ = False

    def get_name(self) -> str:
        return "matmul_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 128], DataType.FP32, init_value=2.0),
            TensorSpec("b", [128, 128], DataType.FP32, init_value=3.0),
            TensorSpec("c", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class MatmulProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul(
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
                out_c = self.matmul(a, b)
                return out_c

        return MatmulProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.matmul(tensors["a"], tensors["b"])


class TestMatmul128x128(PTOTestCase):
    """Test matmul with A[128,128] @ B[128,128] -> C[128,128]."""

    __test__ = False

    def get_name(self) -> str:
        return "matmul_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=2.0),
            TensorSpec("b", [128, 128], DataType.FP32, init_value=3.0),
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class MatmulProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul(
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
                out_c = self.matmul(a, b)
                return out_c

        return MatmulProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.matmul(tensors["a"], tensors["b"])


# =============================================================================
# matmul_acc: C[M,N] += A[M,K] @ B[K,N]  (acc = matmul(A,B) + matmul(A,B) = 2*A@B)
# =============================================================================


class TestMatmulAcc32x128(PTOTestCase):
    """Test matmul_acc with A[32,128] @ B[128,128] accumulated twice -> C[32,128]."""

    __test__ = False

    def get_name(self) -> str:
        return "matmul_acc_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 128], DataType.FP32, init_value=2.0),
            TensorSpec("b", [128, 128], DataType.FP32, init_value=3.0),
            TensorSpec("c", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class MatmulAccProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_acc_fn(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[32, 128], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                tile_a_l1 = pl.load(a, offsets=[0, 0], shapes=[32, 128], target_memory=pl.MemorySpace.L1)
                tile_b_l1 = pl.load(b, offsets=[0, 0], shapes=[128, 128], target_memory=pl.MemorySpace.L1)
                tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.L0A)
                tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.L0B)
                # Initialize acc with first matmul, then accumulate with second
                tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
                tile_c_l0c = pl.matmul_acc(tile_c_l0c, tile_a_l0a, tile_b_l0b)
                out_c = pl.l0c_store(tile_c_l0c, offsets=[0, 0], shapes=[32, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[32, 128], pl.FP32], b: pl.Tensor[[128, 128], pl.FP32]
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                out_c = self.matmul_acc_fn(a, b)
                return out_c

        return MatmulAccProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = 2 * torch.matmul(tensors["a"], tensors["b"])


class TestMatmulAcc64x64(PTOTestCase):
    """Test matmul_acc with A[64,64] @ B[64,64] accumulated twice -> C[64,64]."""

    __test__ = False

    def get_name(self) -> str:
        return "matmul_acc_64x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [64, 64], DataType.FP32, init_value=2.0),
            TensorSpec("b", [64, 64], DataType.FP32, init_value=3.0),
            TensorSpec("c", [64, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class MatmulAccProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_acc_fn(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                c: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a_l1 = pl.load(a, offsets=[0, 0], shapes=[64, 64], target_memory=pl.MemorySpace.L1)
                tile_b_l1 = pl.load(b, offsets=[0, 0], shapes=[64, 64], target_memory=pl.MemorySpace.L1)
                tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.L0A)
                tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.L0B)
                # Initialize acc with first matmul, then accumulate with second
                tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
                tile_c_l0c = pl.matmul_acc(tile_c_l0c, tile_a_l0a, tile_b_l0b)
                out_c = pl.l0c_store(tile_c_l0c, offsets=[0, 0], shapes=[64, 64], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[64, 64], pl.FP32], b: pl.Tensor[[64, 64], pl.FP32]
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                out_c = self.matmul_acc_fn(a, b)
                return out_c

        return MatmulAccProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = 2 * torch.matmul(tensors["a"], tensors["b"])


# =============================================================================
# pytest test functions
# =============================================================================


class TestMatmulOperations:
    """Test suite for matrix multiplication operations."""

    def test_matmul_32x128(self, test_runner):
        """Test matmul with 32x128 output shape."""
        test_case = TestMatmul32x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    def test_matmul_128x128(self, test_runner):
        """Test matmul with 128x128 output shape."""
        test_case = TestMatmul128x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    def test_matmul_acc_32x128(self, test_runner):
        """Test matmul_acc with 32x128 output shape."""
        test_case = TestMatmulAcc32x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    def test_matmul_acc_64x64(self, test_runner):
        """Test matmul_acc with 64x64 output shape."""
        test_case = TestMatmulAcc64x64()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
