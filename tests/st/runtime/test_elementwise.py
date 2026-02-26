# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Runtime tests for tile-based elementwise operations using the PyPTO frontend.

This module defines integration tests for elementwise add and multiply
kernels implemented with the internal PTOTestCase harness, including
variants for different shapes and optimization strategies.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.ir.pass_manager import OptimizationStrategy


class TestTileAdd(PTOTestCase):
    """Test case for tile element-wise addition.

    This test case demonstrates the simplified pattern:
    - Just implement incore function in get_program() and compute_expected()
    - Orchestration function will be auto-generated

    Note: PyPTO requires shape dimensions to be compile-time constants in type
    annotations. For different shapes, create separate test classes (see TestTileAdd64x64).
    """

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_add_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=2.0),
            TensorSpec("b", [128, 128], DataType.FP32, init_value=3.0),
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileAddProgram:
            @pl.function
            def tile_add(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[128, 128])
                tile_b = pl.load(b, offsets=[0, 0], shapes=[128, 128])
                tile_c = pl.add(tile_a, tile_b)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[128, 128], pl.FP32], b: pl.Tensor[[128, 128], pl.FP32]
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_add(a, b)
                return out_c

        return TileAddProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] + tensors["b"]


class TestTileAdd32x128(PTOTestCase):
    """Test tile addition with 32x128 shape."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_add_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 128], DataType.FP32, init_value=2.0),
            TensorSpec("b", [32, 128], DataType.FP32, init_value=3.0),
            TensorSpec("c", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileAddProgram:
            @pl.function
            def tile_add(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                b: pl.Tensor[[32, 128], pl.FP32],
                c: pl.Tensor[[32, 128], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 128])
                tile_b = pl.load(b, offsets=[0, 0], shapes=[32, 128])
                tile_c = pl.add(tile_a, tile_b)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[32, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[32, 128], pl.FP32], b: pl.Tensor[[32, 128], pl.FP32]
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                out_c = self.tile_add(a, b)
                return out_c

        return TileAddProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] + tensors["b"]


class TestTileMul(PTOTestCase):
    """Test case for tile element-wise multiplication (128x128)."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_mul_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            # Method 1: Use Callable to generate random data (different on each run)
            TensorSpec(
                "a",
                [128, 128],
                DataType.FP32,
                init_value=torch.randn,
            ),
            # Method 2: Use scalar value (recommended - simple and serializable)
            TensorSpec("b", [128, 128], DataType.FP32, init_value=3.0),
            # For other methods, see TestCustomArrayInit class examples:
            # - Small arrays can use torch.tensor([[...]])
            # - Identity matrix: torch.eye(n)
            # - Diagonal matrix: torch.diag(torch.tensor([...]))
            # Output tensor: automatically zero-initialized
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileMulProgram:
            @pl.function
            def tile_mul(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[128, 128])
                tile_b = pl.load(b, offsets=[0, 0], shapes=[128, 128])
                tile_c = pl.mul(tile_a, tile_b)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[128, 128], pl.FP32], b: pl.Tensor[[128, 128], pl.FP32]
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_mul(a, b)
                return out_c

        return TileMulProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] * tensors["b"]


class TestTileMul32x128(PTOTestCase):
    """Test tile multiplication with 32x128 shape."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_mul_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "a",
                [32, 128],
                DataType.FP32,
                init_value=torch.randn,
            ),
            TensorSpec("b", [32, 128], DataType.FP32, init_value=3.0),
            TensorSpec("c", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileMulProgram:
            @pl.function
            def tile_mul(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                b: pl.Tensor[[32, 128], pl.FP32],
                c: pl.Tensor[[32, 128], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 128])
                tile_b = pl.load(b, offsets=[0, 0], shapes=[32, 128])
                tile_c = pl.mul(tile_a, tile_b)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[32, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[32, 128], pl.FP32], b: pl.Tensor[[32, 128], pl.FP32]
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                out_c = self.tile_mul(a, b)
                return out_c

        return TileMulProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] * tensors["b"]


class TestTileAddWithPTOAS(TestTileAdd):
    """Test tile add with PTOAS optimization strategy.

    This demonstrates how to use a custom optimization strategy.
    """

    __test__ = False  # Not a pytest test class

    def get_strategy(self):
        return OptimizationStrategy.PTOAS

    def get_name(self) -> str:
        return "tile_add_ptoas_128x128"


class TestTileSub(PTOTestCase):
    """Test case for tile element-wise subtraction (128x128)."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_sub_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=5.0),
            TensorSpec("b", [128, 128], DataType.FP32, init_value=2.0),
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileSubProgram:
            @pl.function
            def tile_sub(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[128, 128])
                tile_b = pl.load(b, offsets=[0, 0], shapes=[128, 128])
                tile_c = pl.sub(tile_a, tile_b)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[128, 128], pl.FP32], b: pl.Tensor[[128, 128], pl.FP32]
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_sub(a, b)
                return out_c

        return TileSubProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] - tensors["b"]


class TestTileSub32x128(PTOTestCase):
    """Test tile subtraction with 32x128 shape."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_sub_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 128], DataType.FP32, init_value=5.0),
            TensorSpec("b", [32, 128], DataType.FP32, init_value=2.0),
            TensorSpec("c", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileSubProgram:
            @pl.function
            def tile_sub(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                b: pl.Tensor[[32, 128], pl.FP32],
                c: pl.Tensor[[32, 128], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 128])
                tile_b = pl.load(b, offsets=[0, 0], shapes=[32, 128])
                tile_c = pl.sub(tile_a, tile_b)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[32, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[32, 128], pl.FP32], b: pl.Tensor[[32, 128], pl.FP32]
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                out_c = self.tile_sub(a, b)
                return out_c

        return TileSubProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] - tensors["b"]


class TestTileDiv(PTOTestCase):
    """Test case for tile element-wise division (128x128)."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_div_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=6.0),
            TensorSpec("b", [128, 128], DataType.FP32, init_value=2.0),
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileDivProgram:
            @pl.function
            def tile_div(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[128, 128])
                tile_b = pl.load(b, offsets=[0, 0], shapes=[128, 128])
                tile_c = pl.div(tile_a, tile_b)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[128, 128], pl.FP32], b: pl.Tensor[[128, 128], pl.FP32]
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_div(a, b)
                return out_c

        return TileDivProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] / tensors["b"]


class TestTileDiv32x128(PTOTestCase):
    """Test tile division with 32x128 shape."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_div_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 128], DataType.FP32, init_value=6.0),
            TensorSpec("b", [32, 128], DataType.FP32, init_value=2.0),
            TensorSpec("c", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileDivProgram:
            @pl.function
            def tile_div(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                b: pl.Tensor[[32, 128], pl.FP32],
                c: pl.Tensor[[32, 128], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 128])
                tile_b = pl.load(b, offsets=[0, 0], shapes=[32, 128])
                tile_c = pl.div(tile_a, tile_b)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[32, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[32, 128], pl.FP32], b: pl.Tensor[[32, 128], pl.FP32]
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                out_c = self.tile_div(a, b)
                return out_c

        return TileDivProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] / tensors["b"]


class TestTileMaximum(PTOTestCase):
    """Test case for tile element-wise maximum (128x128)."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_maximum_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [128, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileMaximumProgram:
            @pl.function
            def tile_maximum(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[128, 128])
                tile_b = pl.load(b, offsets=[0, 0], shapes=[128, 128])
                tile_c = pl.maximum(tile_a, tile_b)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[128, 128], pl.FP32], b: pl.Tensor[[128, 128], pl.FP32]
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_maximum(a, b)
                return out_c

        return TileMaximumProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.maximum(tensors["a"], tensors["b"])


class TestTileMaximum32x128(PTOTestCase):
    """Test tile maximum with 32x128 shape."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_maximum_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [32, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileMaximumProgram:
            @pl.function
            def tile_maximum(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                b: pl.Tensor[[32, 128], pl.FP32],
                c: pl.Tensor[[32, 128], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 128])
                tile_b = pl.load(b, offsets=[0, 0], shapes=[32, 128])
                tile_c = pl.maximum(tile_a, tile_b)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[32, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[32, 128], pl.FP32], b: pl.Tensor[[32, 128], pl.FP32]
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                out_c = self.tile_maximum(a, b)
                return out_c

        return TileMaximumProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.maximum(tensors["a"], tensors["b"])


class TestTileMinimum(PTOTestCase):
    """Test case for tile element-wise minimum (128x128)."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_minimum_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [128, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileMinimumProgram:
            @pl.function
            def tile_minimum(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[128, 128])
                tile_b = pl.load(b, offsets=[0, 0], shapes=[128, 128])
                tile_c = pl.minimum(tile_a, tile_b)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[128, 128], pl.FP32], b: pl.Tensor[[128, 128], pl.FP32]
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_minimum(a, b)
                return out_c

        return TileMinimumProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.minimum(tensors["a"], tensors["b"])


class TestTileMinimum32x128(PTOTestCase):
    """Test tile minimum with 32x128 shape."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_minimum_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [32, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileMinimumProgram:
            @pl.function
            def tile_minimum(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                b: pl.Tensor[[32, 128], pl.FP32],
                c: pl.Tensor[[32, 128], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 128])
                tile_b = pl.load(b, offsets=[0, 0], shapes=[32, 128])
                tile_c = pl.minimum(tile_a, tile_b)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[32, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[32, 128], pl.FP32], b: pl.Tensor[[32, 128], pl.FP32]
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                out_c = self.tile_minimum(a, b)
                return out_c

        return TileMinimumProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.minimum(tensors["a"], tensors["b"])


class TestTileAdds(PTOTestCase):
    """Test case for tile-scalar addition (128x128)."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_adds_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=3.0),
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileAddsProgram:
            @pl.function
            def tile_adds(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[128, 128])
                tile_c = pl.adds(tile_a, 2.0)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_adds(a)
                return out_c

        return TileAddsProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] + 2.0


class TestTileAdds32x128(PTOTestCase):
    """Test tile-scalar addition with 32x128 shape."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_adds_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 128], DataType.FP32, init_value=3.0),
            TensorSpec("c", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileAddsProgram:
            @pl.function
            def tile_adds(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                c: pl.Tensor[[32, 128], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 128])
                tile_c = pl.adds(tile_a, 2.0)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[32, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[32, 128], pl.FP32]) -> pl.Tensor[[32, 128], pl.FP32]:
                out_c = self.tile_adds(a)
                return out_c

        return TileAddsProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] + 2.0


class TestTileSubs(PTOTestCase):
    """Test case for tile-scalar subtraction (128x128)."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_subs_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=5.0),
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileSubsProgram:
            @pl.function
            def tile_subs(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[128, 128])
                tile_c = pl.subs(tile_a, 2.0)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_subs(a)
                return out_c

        return TileSubsProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] - 2.0


class TestTileSubs32x128(PTOTestCase):
    """Test tile-scalar subtraction with 32x128 shape."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_subs_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 128], DataType.FP32, init_value=5.0),
            TensorSpec("c", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileSubsProgram:
            @pl.function
            def tile_subs(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                c: pl.Tensor[[32, 128], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 128])
                tile_c = pl.subs(tile_a, 2.0)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[32, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[32, 128], pl.FP32]) -> pl.Tensor[[32, 128], pl.FP32]:
                out_c = self.tile_subs(a)
                return out_c

        return TileSubsProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] - 2.0


class TestTileMuls(PTOTestCase):
    """Test case for tile-scalar multiplication (128x128)."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_muls_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=3.0),
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileMulsProgram:
            @pl.function
            def tile_muls(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[128, 128])
                tile_c = pl.muls(tile_a, 2.0)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_muls(a)
                return out_c

        return TileMulsProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] * 2.0


class TestTileMuls32x128(PTOTestCase):
    """Test tile-scalar multiplication with 32x128 shape."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_muls_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 128], DataType.FP32, init_value=3.0),
            TensorSpec("c", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileMulsProgram:
            @pl.function
            def tile_muls(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                c: pl.Tensor[[32, 128], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 128])
                tile_c = pl.muls(tile_a, 2.0)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[32, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[32, 128], pl.FP32]) -> pl.Tensor[[32, 128], pl.FP32]:
                out_c = self.tile_muls(a)
                return out_c

        return TileMulsProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] * 2.0


class TestTileDivs(PTOTestCase):
    """Test case for tile-scalar division (128x128)."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_divs_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=6.0),
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileDivsProgram:
            @pl.function
            def tile_divs(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[128, 128])
                tile_c = pl.divs(tile_a, 2.0)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_divs(a)
                return out_c

        return TileDivsProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] / 2.0


class TestTileDivs32x128(PTOTestCase):
    """Test tile-scalar division with 32x128 shape."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_divs_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 128], DataType.FP32, init_value=6.0),
            TensorSpec("c", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileDivsProgram:
            @pl.function
            def tile_divs(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                c: pl.Tensor[[32, 128], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 128])
                tile_c = pl.divs(tile_a, 2.0)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[32, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[32, 128], pl.FP32]) -> pl.Tensor[[32, 128], pl.FP32]:
                out_c = self.tile_divs(a)
                return out_c

        return TileDivsProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] / 2.0


class TestTileCmp(PTOTestCase):
    """Test case for tile element-wise comparison, GT mode (128x128)."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_cmp_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [128, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileCmpProgram:
            @pl.function
            def tile_cmp(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[128, 128])
                tile_b = pl.load(b, offsets=[0, 0], shapes=[128, 128])
                tile_c = pl.cmp(tile_a, tile_b, cmp_type=4)
                tile_c_fp32 = pl.cast(tile_c, target_type=pl.FP32)
                out_c = pl.store(tile_c_fp32, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[128, 128], pl.FP32], b: pl.Tensor[[128, 128], pl.FP32]
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_cmp(a, b)
                return out_c

        return TileCmpProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = (tensors["a"] > tensors["b"]).to(torch.float32)


class TestTileCmp32x128(PTOTestCase):
    """Test tile comparison with 32x128 shape, GT mode."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_cmp_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [32, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileCmpProgram:
            @pl.function
            def tile_cmp(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                b: pl.Tensor[[32, 128], pl.FP32],
                c: pl.Tensor[[32, 128], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 128])
                tile_b = pl.load(b, offsets=[0, 0], shapes=[32, 128])
                tile_c = pl.cmp(tile_a, tile_b, cmp_type=4)
                tile_c_fp32 = pl.cast(tile_c, target_type=pl.FP32)
                out_c = pl.store(tile_c_fp32, offsets=[0, 0], shapes=[32, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[32, 128], pl.FP32], b: pl.Tensor[[32, 128], pl.FP32]
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                out_c = self.tile_cmp(a, b)
                return out_c

        return TileCmpProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = (tensors["a"] > tensors["b"]).to(torch.float32)


class TestTileCmps(PTOTestCase):
    """Test case for tile-scalar comparison, GT mode (128x128)."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_cmps_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileCmpsProgram:
            @pl.function
            def tile_cmps(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[128, 128])
                tile_c = pl.cmps(tile_a, 0.0, cmp_type=4)
                tile_c_fp32 = pl.cast(tile_c, target_type=pl.FP32)
                out_c = pl.store(tile_c_fp32, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_cmps(a)
                return out_c

        return TileCmpsProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = (tensors["a"] > 0.0).to(torch.float32)


class TestTileCmps32x128(PTOTestCase):
    """Test tile-scalar comparison with 32x128 shape, GT mode."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_cmps_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileCmpsProgram:
            @pl.function
            def tile_cmps(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                c: pl.Tensor[[32, 128], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 128])
                tile_c = pl.cmps(tile_a, 0.0, cmp_type=4)
                tile_c_fp32 = pl.cast(tile_c, target_type=pl.FP32)
                out_c = pl.store(tile_c_fp32, offsets=[0, 0], shapes=[32, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[32, 128], pl.FP32]) -> pl.Tensor[[32, 128], pl.FP32]:
                out_c = self.tile_cmps(a)
                return out_c

        return TileCmpsProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = (tensors["a"] > 0.0).to(torch.float32)


class TestCustomArrayInit(PTOTestCase):
    """Test case demonstrating custom array initialization patterns."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "custom_array_init"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            # Small array: custom values (will be serialized)
            TensorSpec(
                "small",
                [3, 3],
                DataType.FP32,
                init_value=torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32),
            ),
            # Identity matrix
            TensorSpec("identity", [4, 4], DataType.FP32, init_value=torch.eye(4, dtype=torch.float32)),
            # Constant array (optimized to torch.full)
            TensorSpec("constant", [5, 5], DataType.FP32, init_value=torch.ones((5, 5)) * 3.14),
            # Diagonal matrix (small arrays will be serialized)
            TensorSpec(
                "diagonal",
                [3, 3],
                DataType.FP32,
                init_value=torch.diag(torch.tensor([1, 2, 3], dtype=torch.float32)),
            ),
            # Output
            TensorSpec("out", [3, 3], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        # Placeholder - this test is just for demonstrating array initialization
        return None

    def compute_expected(self, tensors, params=None):
        # Simple example: copy small array to output
        tensors["out"][:] = tensors["small"][:3, :3]


# =============================================================================
# pytest test functions
# =============================================================================


class TestElementwiseOperations:
    """Test suite for elementwise operations."""

    def test_tile_add_32x128(self, test_runner):
        """Test tile addition with 32x128 shape."""
        test_case = TestTileAdd32x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 32x128: {result.error}"

    def test_tile_add_128x128(self, test_runner):
        """Test tile addition with 128x128 shape."""
        test_case = TestTileAdd()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"

    def test_tile_mul_32x128(self, test_runner):
        """Test tile multiplication with 32x128 shape."""
        test_case = TestTileMul32x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 32x128: {result.error}"

    def test_tile_mul_128x128(self, test_runner):
        """Test tile multiplication with 128x128 shape."""
        test_case = TestTileMul()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"

    def test_tile_sub_128x128(self, test_runner):
        """Test tile subtraction with 128x128 shape."""
        test_case = TestTileSub()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"

    def test_tile_sub_32x128(self, test_runner):
        """Test tile subtraction with 32x128 shape."""
        test_case = TestTileSub32x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 32x128: {result.error}"

    def test_tile_div_128x128(self, test_runner):
        """Test tile division with 128x128 shape."""
        test_case = TestTileDiv()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"

    def test_tile_div_32x128(self, test_runner):
        """Test tile division with 32x128 shape."""
        test_case = TestTileDiv32x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 32x128: {result.error}"

    def test_tile_maximum_128x128(self, test_runner):
        """Test tile element-wise maximum with 128x128 shape."""
        test_case = TestTileMaximum()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"

    def test_tile_maximum_32x128(self, test_runner):
        """Test tile element-wise maximum with 32x128 shape."""
        test_case = TestTileMaximum32x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 32x128: {result.error}"

    def test_tile_minimum_128x128(self, test_runner):
        """Test tile element-wise minimum with 128x128 shape."""
        test_case = TestTileMinimum()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"

    def test_tile_minimum_32x128(self, test_runner):
        """Test tile element-wise minimum with 32x128 shape."""
        test_case = TestTileMinimum32x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 32x128: {result.error}"

    def test_tile_adds_128x128(self, test_runner):
        """Test tile-scalar addition with 128x128 shape."""
        test_case = TestTileAdds()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"

    def test_tile_adds_32x128(self, test_runner):
        """Test tile-scalar addition with 32x128 shape."""
        test_case = TestTileAdds32x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 32x128: {result.error}"

    def test_tile_subs_128x128(self, test_runner):
        """Test tile-scalar subtraction with 128x128 shape."""
        test_case = TestTileSubs()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"

    def test_tile_subs_32x128(self, test_runner):
        """Test tile-scalar subtraction with 32x128 shape."""
        test_case = TestTileSubs32x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 32x128: {result.error}"

    def test_tile_muls_128x128(self, test_runner):
        """Test tile-scalar multiplication with 128x128 shape."""
        test_case = TestTileMuls()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"

    def test_tile_muls_32x128(self, test_runner):
        """Test tile-scalar multiplication with 32x128 shape."""
        test_case = TestTileMuls32x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 32x128: {result.error}"

    def test_tile_divs_128x128(self, test_runner):
        """Test tile-scalar division with 128x128 shape."""
        test_case = TestTileDivs()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 128x128: {result.error}"

    def test_tile_divs_32x128(self, test_runner):
        """Test tile-scalar division with 32x128 shape."""
        test_case = TestTileDivs32x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for 32x128: {result.error}"

    @pytest.mark.skip(reason="PTOAS optimization strategy has calculation issues - needs investigation")
    def test_tile_add_ptoas_strategy(self, test_runner):
        """Test tile addition with PTOAS optimization strategy."""
        test_case = TestTileAddWithPTOAS()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
