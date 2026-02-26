# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Runtime tests for tile-based unary operations using the PyPTO frontend.

This module defines integration tests for unary kernels (log, abs, relu, exp,
sqrt, neg) implemented with the internal PTOTestCase harness, including
variants for different shapes (128x128 and 32x128).
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec


class TestTileLog(PTOTestCase):
    """Test case for tile-wise natural logarithm (128x128)."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_log_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=2.0),
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileLogProgram:
            @pl.function
            def tile_log(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.block.load(a, offsets=[0, 0], shapes=[128, 128])
                tile_c = pl.log(tile_a)
                out_c = pl.block.store(tile_c, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[128, 128], pl.FP32]
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_log(a)
                return out_c

        return TileLogProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.log(tensors["a"])


class TestTileLog32x128(PTOTestCase):
    """Test case for tile-wise natural logarithm (32x128)."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_log_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 128], DataType.FP32, init_value=2.0),
            TensorSpec("c", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileLogProgram:
            @pl.function
            def tile_log(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                c: pl.Tensor[[32, 128], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                tile_a = pl.block.load(a, offsets=[0, 0], shapes=[32, 128])
                tile_c = pl.log(tile_a)
                out_c = pl.block.store(tile_c, offsets=[0, 0], shapes=[32, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[32, 128], pl.FP32]
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                out_c = self.tile_log(a)
                return out_c

        return TileLogProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.log(tensors["a"])


class TestTileAbs(PTOTestCase):
    """Test case for tile-wise absolute value (128x128)."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_abs_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileAbsProgram:
            @pl.function
            def tile_abs(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.block.load(a, offsets=[0, 0], shapes=[128, 128])
                tile_c = pl.abs(tile_a)
                out_c = pl.block.store(tile_c, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[128, 128], pl.FP32]
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_abs(a)
                return out_c

        return TileAbsProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.abs(tensors["a"])


class TestTileAbs32x128(PTOTestCase):
    """Test case for tile-wise absolute value (32x128)."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_abs_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileAbsProgram:
            @pl.function
            def tile_abs(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                c: pl.Tensor[[32, 128], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                tile_a = pl.block.load(a, offsets=[0, 0], shapes=[32, 128])
                tile_c = pl.abs(tile_a)
                out_c = pl.block.store(tile_c, offsets=[0, 0], shapes=[32, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[32, 128], pl.FP32]
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                out_c = self.tile_abs(a)
                return out_c

        return TileAbsProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.abs(tensors["a"])


class TestTileRelu(PTOTestCase):
    """Test case for tile-wise ReLU activation (128x128)."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_relu_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileReluProgram:
            @pl.function
            def tile_relu(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.block.load(a, offsets=[0, 0], shapes=[128, 128])
                tile_c = pl.relu(tile_a)
                out_c = pl.block.store(tile_c, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[128, 128], pl.FP32]
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_relu(a)
                return out_c

        return TileReluProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.relu(tensors["a"])


class TestTileRelu32x128(PTOTestCase):
    """Test case for tile-wise ReLU activation (32x128)."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_relu_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileReluProgram:
            @pl.function
            def tile_relu(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                c: pl.Tensor[[32, 128], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                tile_a = pl.block.load(a, offsets=[0, 0], shapes=[32, 128])
                tile_c = pl.relu(tile_a)
                out_c = pl.block.store(tile_c, offsets=[0, 0], shapes=[32, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[32, 128], pl.FP32]
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                out_c = self.tile_relu(a)
                return out_c

        return TileReluProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.relu(tensors["a"])


class TestTileExp(PTOTestCase):
    """Test case for tile-wise exponential (128x128)."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_exp_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=1.0),
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileExpProgram:
            @pl.function
            def tile_exp(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.block.load(a, offsets=[0, 0], shapes=[128, 128])
                tile_c = pl.exp(tile_a)
                out_c = pl.block.store(tile_c, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[128, 128], pl.FP32]
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_exp(a)
                return out_c

        return TileExpProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.exp(tensors["a"])


class TestTileExp32x128(PTOTestCase):
    """Test case for tile-wise exponential (32x128)."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_exp_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 128], DataType.FP32, init_value=1.0),
            TensorSpec("c", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileExpProgram:
            @pl.function
            def tile_exp(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                c: pl.Tensor[[32, 128], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                tile_a = pl.block.load(a, offsets=[0, 0], shapes=[32, 128])
                tile_c = pl.exp(tile_a)
                out_c = pl.block.store(tile_c, offsets=[0, 0], shapes=[32, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[32, 128], pl.FP32]
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                out_c = self.tile_exp(a)
                return out_c

        return TileExpProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.exp(tensors["a"])


class TestTileSqrt(PTOTestCase):
    """Test case for tile-wise square root (128x128)."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_sqrt_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=4.0),
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileSqrtProgram:
            @pl.function
            def tile_sqrt(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.block.load(a, offsets=[0, 0], shapes=[128, 128])
                tile_c = pl.sqrt(tile_a)
                out_c = pl.block.store(tile_c, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[128, 128], pl.FP32]
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_sqrt(a)
                return out_c

        return TileSqrtProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.sqrt(tensors["a"])


class TestTileSqrt32x128(PTOTestCase):
    """Test case for tile-wise square root (32x128)."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_sqrt_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 128], DataType.FP32, init_value=4.0),
            TensorSpec("c", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileSqrtProgram:
            @pl.function
            def tile_sqrt(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                c: pl.Tensor[[32, 128], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                tile_a = pl.block.load(a, offsets=[0, 0], shapes=[32, 128])
                tile_c = pl.sqrt(tile_a)
                out_c = pl.block.store(tile_c, offsets=[0, 0], shapes=[32, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[32, 128], pl.FP32]
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                out_c = self.tile_sqrt(a)
                return out_c

        return TileSqrtProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.sqrt(tensors["a"])


class TestTileNeg(PTOTestCase):
    """Test case for tile-wise negation (128x128)."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_neg_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=2.0),
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileNegProgram:
            @pl.function
            def tile_neg(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.block.load(a, offsets=[0, 0], shapes=[128, 128])
                tile_c = pl.neg(tile_a)
                out_c = pl.block.store(tile_c, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[128, 128], pl.FP32]
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_neg(a)
                return out_c

        return TileNegProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = -tensors["a"]


class TestTileNeg32x128(PTOTestCase):
    """Test case for tile-wise negation (32x128)."""

    __test__ = False  # Not a pytest test class

    def get_name(self) -> str:
        return "tile_neg_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 128], DataType.FP32, init_value=2.0),
            TensorSpec("c", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileNegProgram:
            @pl.function
            def tile_neg(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                c: pl.Tensor[[32, 128], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                tile_a = pl.block.load(a, offsets=[0, 0], shapes=[32, 128])
                tile_c = pl.neg(tile_a)
                out_c = pl.block.store(tile_c, offsets=[0, 0], shapes=[32, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[32, 128], pl.FP32]
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                out_c = self.tile_neg(a)
                return out_c

        return TileNegProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = -tensors["a"]


# =============================================================================
# pytest test suite
# =============================================================================


class TestLogOperations:
    """Test suite for tile-wise natural logarithm."""

    def test_tile_log_128x128(self, test_runner):
        """Test tile-wise natural logarithm (128x128)."""
        test_case = TestTileLog()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    def test_tile_log_32x128(self, test_runner):
        """Test tile-wise natural logarithm (32x128)."""
        test_case = TestTileLog32x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"


class TestAbsOperations:
    """Test suite for tile-wise absolute value."""

    def test_tile_abs_128x128(self, test_runner):
        """Test tile-wise absolute value (128x128)."""
        test_case = TestTileAbs()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    def test_tile_abs_32x128(self, test_runner):
        """Test tile-wise absolute value (32x128)."""
        test_case = TestTileAbs32x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"


class TestReluOperations:
    """Test suite for tile-wise ReLU activation."""

    def test_tile_relu_128x128(self, test_runner):
        """Test tile-wise ReLU activation (128x128)."""
        test_case = TestTileRelu()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    def test_tile_relu_32x128(self, test_runner):
        """Test tile-wise ReLU activation (32x128)."""
        test_case = TestTileRelu32x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"


class TestExpOperations:
    """Test suite for tile-wise exponential."""

    def test_tile_exp_128x128(self, test_runner):
        """Test tile-wise exponential (128x128)."""
        test_case = TestTileExp()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    def test_tile_exp_32x128(self, test_runner):
        """Test tile-wise exponential (32x128)."""
        test_case = TestTileExp32x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"


class TestSqrtOperations:
    """Test suite for tile-wise square root."""

    def test_tile_sqrt_128x128(self, test_runner):
        """Test tile-wise square root (128x128)."""
        test_case = TestTileSqrt()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    def test_tile_sqrt_32x128(self, test_runner):
        """Test tile-wise square root (32x128)."""
        test_case = TestTileSqrt32x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"


class TestNegOperations:
    """Test suite for tile-wise negation."""

    def test_tile_neg_128x128(self, test_runner):
        """Test tile-wise negation (128x128)."""
        test_case = TestTileNeg()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    def test_tile_neg_32x128(self, test_runner):
        """Test tile-wise negation (32x128)."""
        test_case = TestTileNeg32x128()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
