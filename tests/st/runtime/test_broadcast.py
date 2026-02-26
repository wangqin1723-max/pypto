# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Runtime tests for broadcast operations using the PyPTO frontend.

Covers all broadcast ops registered in src/ir/op/block_ops/broadcast.cpp:
- row_expand:     broadcast first col of each row across all cols
- row_expand_add: tile + row_vec (row-wise broadcast)
- row_expand_sub: tile - row_vec (row-wise broadcast)
- row_expand_mul: tile * row_vec (row-wise broadcast)
- row_expand_div: tile / row_vec (row-wise broadcast)
- col_expand:     broadcast col_vec [1,N] to tile shape [M,N]
- col_expand_mul: tile * col_vec (col-wise broadcast)
- col_expand_div: tile / col_vec (col-wise broadcast)
- col_expand_sub: tile - col_vec (col-wise broadcast)
- expands:        broadcast scalar to tile shape

Each op has a 32x128 (non-square) and a 128x128 (square) test case.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

# =============================================================================
# row_expand: broadcast first element of each row across all columns
# =============================================================================


class TestTileRowExpand32x128(PTOTestCase):
    """Test row_expand with 32x128 shape: dst[i,j] = src[i,0]."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_row_expand_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileRowExpandProgram:
            @pl.function
            def tile_row_expand(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                c: pl.Tensor[[32, 128], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 128])
                tile_c = pl.row_expand(tile_a)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[32, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[32, 128], pl.FP32]) -> pl.Tensor[[32, 128], pl.FP32]:
                out_c = self.tile_row_expand(a)
                return out_c

        return TileRowExpandProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"][:, 0:1]


class TestTileRowExpand128x128(PTOTestCase):
    """Test row_expand with 128x128 shape: dst[i,j] = src[i,0]."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_row_expand_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileRowExpandProgram:
            @pl.function
            def tile_row_expand(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[128, 128])
                tile_c = pl.row_expand(tile_a)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_row_expand(a)
                return out_c

        return TileRowExpandProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"][:, 0:1]


# =============================================================================
# row_expand_add: tile + row_vec (row-wise broadcast addition)
# =============================================================================


class TestTileRowExpandAdd32x128(PTOTestCase):
    """Test row_expand_add with 32x128 shape."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_row_expand_add_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("row_vec", [32, 1], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileRowExpandAddProgram:
            @pl.function
            def tile_row_expand_add(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                row_vec: pl.Tensor[[32, 1], pl.FP32],
                c: pl.Tensor[[32, 128], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 128])
                tile_row = pl.load(row_vec, offsets=[0, 0], shapes=[32, 1])
                tile_c = pl.row_expand_add(tile_a, tile_row)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[32, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                row_vec: pl.Tensor[[32, 1], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                out_c = self.tile_row_expand_add(a, row_vec)
                return out_c

        return TileRowExpandAddProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] + tensors["row_vec"]


class TestTileRowExpandAdd128x128(PTOTestCase):
    """Test row_expand_add with 128x128 shape."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_row_expand_add_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("row_vec", [128, 1], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileRowExpandAddProgram:
            @pl.function
            def tile_row_expand_add(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                row_vec: pl.Tensor[[128, 1], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[128, 128])
                tile_row = pl.load(row_vec, offsets=[0, 0], shapes=[128, 1])
                tile_c = pl.row_expand_add(tile_a, tile_row)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                row_vec: pl.Tensor[[128, 1], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_row_expand_add(a, row_vec)
                return out_c

        return TileRowExpandAddProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] + tensors["row_vec"]


# =============================================================================
# row_expand_sub: tile - row_vec (row-wise broadcast subtraction)
# =============================================================================


class TestTileRowExpandSub32x128(PTOTestCase):
    """Test row_expand_sub with 32x128 shape."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_row_expand_sub_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("row_vec", [32, 1], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileRowExpandSubProgram:
            @pl.function
            def tile_row_expand_sub(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                row_vec: pl.Tensor[[32, 1], pl.FP32],
                c: pl.Tensor[[32, 128], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 128])
                tile_row = pl.load(row_vec, offsets=[0, 0], shapes=[32, 1])
                tile_c = pl.row_expand_sub(tile_a, tile_row)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[32, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                row_vec: pl.Tensor[[32, 1], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                out_c = self.tile_row_expand_sub(a, row_vec)
                return out_c

        return TileRowExpandSubProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] - tensors["row_vec"]


class TestTileRowExpandSub128x128(PTOTestCase):
    """Test row_expand_sub with 128x128 shape."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_row_expand_sub_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("row_vec", [128, 1], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileRowExpandSubProgram:
            @pl.function
            def tile_row_expand_sub(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                row_vec: pl.Tensor[[128, 1], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[128, 128])
                tile_row = pl.load(row_vec, offsets=[0, 0], shapes=[128, 1])
                tile_c = pl.row_expand_sub(tile_a, tile_row)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                row_vec: pl.Tensor[[128, 1], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_row_expand_sub(a, row_vec)
                return out_c

        return TileRowExpandSubProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] - tensors["row_vec"]


# =============================================================================
# row_expand_mul: tile * row_vec (row-wise broadcast multiplication)
# =============================================================================


class TestTileRowExpandMul32x128(PTOTestCase):
    """Test row_expand_mul with 32x128 shape."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_row_expand_mul_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("row_vec", [32, 1], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileRowExpandMulProgram:
            @pl.function
            def tile_row_expand_mul(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                row_vec: pl.Tensor[[32, 1], pl.FP32],
                c: pl.Tensor[[32, 128], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 128])
                tile_row = pl.load(row_vec, offsets=[0, 0], shapes=[32, 1])
                tile_c = pl.row_expand_mul(tile_a, tile_row)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[32, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                row_vec: pl.Tensor[[32, 1], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                out_c = self.tile_row_expand_mul(a, row_vec)
                return out_c

        return TileRowExpandMulProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] * tensors["row_vec"]


class TestTileRowExpandMul128x128(PTOTestCase):
    """Test row_expand_mul with 128x128 shape."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_row_expand_mul_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("row_vec", [128, 1], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileRowExpandMulProgram:
            @pl.function
            def tile_row_expand_mul(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                row_vec: pl.Tensor[[128, 1], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[128, 128])
                tile_row = pl.load(row_vec, offsets=[0, 0], shapes=[128, 1])
                tile_c = pl.row_expand_mul(tile_a, tile_row)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                row_vec: pl.Tensor[[128, 1], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_row_expand_mul(a, row_vec)
                return out_c

        return TileRowExpandMulProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] * tensors["row_vec"]


# =============================================================================
# row_expand_div: tile / row_vec (row-wise broadcast division)
# =============================================================================


class TestTileRowExpandDiv32x128(PTOTestCase):
    """Test row_expand_div with 32x128 shape."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_row_expand_div_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("row_vec", [32, 1], DataType.FP32, init_value=2.0),  # avoid division by zero
            TensorSpec("c", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileRowExpandDivProgram:
            @pl.function
            def tile_row_expand_div(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                row_vec: pl.Tensor[[32, 1], pl.FP32],
                c: pl.Tensor[[32, 128], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 128])
                tile_row = pl.load(row_vec, offsets=[0, 0], shapes=[32, 1])
                tile_c = pl.row_expand_div(tile_a, tile_row)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[32, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                row_vec: pl.Tensor[[32, 1], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                out_c = self.tile_row_expand_div(a, row_vec)
                return out_c

        return TileRowExpandDivProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] / tensors["row_vec"]


class TestTileRowExpandDiv128x128(PTOTestCase):
    """Test row_expand_div with 128x128 shape."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_row_expand_div_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("row_vec", [128, 1], DataType.FP32, init_value=2.0),  # avoid division by zero
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileRowExpandDivProgram:
            @pl.function
            def tile_row_expand_div(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                row_vec: pl.Tensor[[128, 1], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[128, 128])
                tile_row = pl.load(row_vec, offsets=[0, 0], shapes=[128, 1])
                tile_c = pl.row_expand_div(tile_a, tile_row)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                row_vec: pl.Tensor[[128, 1], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_row_expand_div(a, row_vec)
                return out_c

        return TileRowExpandDivProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] / tensors["row_vec"]


# =============================================================================
# col_expand: broadcast col_vec [1,N] to tile shape [M,N]
# =============================================================================


class TestTileColExpand32x128(PTOTestCase):
    """Test col_expand with 32x128 shape: expand [1,128] to [32,128]."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_col_expand_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 128], DataType.FP32, init_value=1.0),  # shape reference only
            TensorSpec("col_vec", [1, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileColExpandProgram:
            @pl.function
            def tile_col_expand(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                col_vec: pl.Tensor[[1, 128], pl.FP32],
                c: pl.Tensor[[32, 128], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 128])
                tile_col = pl.load(col_vec, offsets=[0, 0], shapes=[1, 128])
                tile_c = pl.col_expand(tile_a, tile_col)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[32, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                col_vec: pl.Tensor[[1, 128], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                out_c = self.tile_col_expand(a, col_vec)
                return out_c

        return TileColExpandProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["col_vec"]  # [1,128] broadcasts to [32,128]


class TestTileColExpand128x128(PTOTestCase):
    """Test col_expand with 128x128 shape: expand [1,128] to [128,128]."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_col_expand_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=1.0),  # shape reference only
            TensorSpec("col_vec", [1, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileColExpandProgram:
            @pl.function
            def tile_col_expand(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                col_vec: pl.Tensor[[1, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[128, 128])
                tile_col = pl.load(col_vec, offsets=[0, 0], shapes=[1, 128])
                tile_c = pl.col_expand(tile_a, tile_col)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                col_vec: pl.Tensor[[1, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_col_expand(a, col_vec)
                return out_c

        return TileColExpandProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["col_vec"]  # [1,128] broadcasts to [128,128]


# =============================================================================
# col_expand_mul: tile * col_vec (col-wise broadcast multiplication)
# =============================================================================


class TestTileColExpandMul32x128(PTOTestCase):
    """Test col_expand_mul with 32x128 shape."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_col_expand_mul_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("col_vec", [1, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileColExpandMulProgram:
            @pl.function
            def tile_col_expand_mul(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                col_vec: pl.Tensor[[1, 128], pl.FP32],
                c: pl.Tensor[[32, 128], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 128])
                tile_col = pl.load(col_vec, offsets=[0, 0], shapes=[1, 128])
                tile_c = pl.col_expand_mul(tile_a, tile_col)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[32, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                col_vec: pl.Tensor[[1, 128], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                out_c = self.tile_col_expand_mul(a, col_vec)
                return out_c

        return TileColExpandMulProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] * tensors["col_vec"]


class TestTileColExpandMul128x128(PTOTestCase):
    """Test col_expand_mul with 128x128 shape."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_col_expand_mul_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("col_vec", [1, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileColExpandMulProgram:
            @pl.function
            def tile_col_expand_mul(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                col_vec: pl.Tensor[[1, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[128, 128])
                tile_col = pl.load(col_vec, offsets=[0, 0], shapes=[1, 128])
                tile_c = pl.col_expand_mul(tile_a, tile_col)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                col_vec: pl.Tensor[[1, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_col_expand_mul(a, col_vec)
                return out_c

        return TileColExpandMulProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] * tensors["col_vec"]


# =============================================================================
# col_expand_div: tile / col_vec (col-wise broadcast division)
# =============================================================================


class TestTileColExpandDiv32x128(PTOTestCase):
    """Test col_expand_div with 32x128 shape."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_col_expand_div_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("col_vec", [1, 128], DataType.FP32, init_value=2.0),  # avoid division by zero
            TensorSpec("c", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileColExpandDivProgram:
            @pl.function
            def tile_col_expand_div(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                col_vec: pl.Tensor[[1, 128], pl.FP32],
                c: pl.Tensor[[32, 128], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 128])
                tile_col = pl.load(col_vec, offsets=[0, 0], shapes=[1, 128])
                tile_c = pl.col_expand_div(tile_a, tile_col)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[32, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                col_vec: pl.Tensor[[1, 128], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                out_c = self.tile_col_expand_div(a, col_vec)
                return out_c

        return TileColExpandDivProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] / tensors["col_vec"]


class TestTileColExpandDiv128x128(PTOTestCase):
    """Test col_expand_div with 128x128 shape."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_col_expand_div_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("col_vec", [1, 128], DataType.FP32, init_value=2.0),  # avoid division by zero
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileColExpandDivProgram:
            @pl.function
            def tile_col_expand_div(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                col_vec: pl.Tensor[[1, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[128, 128])
                tile_col = pl.load(col_vec, offsets=[0, 0], shapes=[1, 128])
                tile_c = pl.col_expand_div(tile_a, tile_col)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                col_vec: pl.Tensor[[1, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_col_expand_div(a, col_vec)
                return out_c

        return TileColExpandDivProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] / tensors["col_vec"]


# =============================================================================
# col_expand_sub: tile - col_vec (col-wise broadcast subtraction)
# =============================================================================


class TestTileColExpandSub32x128(PTOTestCase):
    """Test col_expand_sub with 32x128 shape."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_col_expand_sub_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("col_vec", [1, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileColExpandSubProgram:
            @pl.function
            def tile_col_expand_sub(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                col_vec: pl.Tensor[[1, 128], pl.FP32],
                c: pl.Tensor[[32, 128], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 128])
                tile_col = pl.load(col_vec, offsets=[0, 0], shapes=[1, 128])
                tile_c = pl.col_expand_sub(tile_a, tile_col)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[32, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                col_vec: pl.Tensor[[1, 128], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                out_c = self.tile_col_expand_sub(a, col_vec)
                return out_c

        return TileColExpandSubProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] - tensors["col_vec"]


class TestTileColExpandSub128x128(PTOTestCase):
    """Test col_expand_sub with 128x128 shape."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_col_expand_sub_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("col_vec", [1, 128], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileColExpandSubProgram:
            @pl.function
            def tile_col_expand_sub(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                col_vec: pl.Tensor[[1, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[128, 128])
                tile_col = pl.load(col_vec, offsets=[0, 0], shapes=[1, 128])
                tile_c = pl.col_expand_sub(tile_a, tile_col)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                col_vec: pl.Tensor[[1, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_col_expand_sub(a, col_vec)
                return out_c

        return TileColExpandSubProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] - tensors["col_vec"]


# =============================================================================
# expands: broadcast scalar value to tile shape
# =============================================================================


class TestTileExpandScalar32x128(PTOTestCase):
    """Test expands with 32x128 shape: fill tile with scalar 2.0."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_expands_32x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 128], DataType.FP32, init_value=1.0),  # shape reference only
            TensorSpec("c", [32, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileExpandScalarProgram:
            @pl.function
            def tile_expands(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                c: pl.Tensor[[32, 128], pl.FP32],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 128])
                tile_c = pl.expands(tile_a, 2.0)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[32, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[32, 128], pl.FP32]) -> pl.Tensor[[32, 128], pl.FP32]:
                out_c = self.tile_expands(a)
                return out_c

        return TileExpandScalarProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = 2.0


class TestTileExpandScalar128x128(PTOTestCase):
    """Test expands with 128x128 shape: fill tile with scalar 2.0."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_expands_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=1.0),  # shape reference only
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TileExpandScalarProgram:
            @pl.function
            def tile_expands(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[128, 128])
                tile_c = pl.expands(tile_a, 2.0)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_expands(a)
                return out_c

        return TileExpandScalarProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = 2.0


# =============================================================================
# pytest test functions
# =============================================================================


class TestBroadcastOperations:
    """Test suite for all broadcast operations."""

    # --- row_expand ---
    # def test_tile_row_expand_32x128(self, test_runner):
    #     """Test row_expand with 32x128 shape."""
    #     result = test_runner.run(TestTileRowExpand32x128())
    #     assert result.passed, f"Test failed: {result.error}"

    # def test_tile_row_expand_128x128(self, test_runner):
    #     """Test row_expand with 128x128 shape."""
    #     result = test_runner.run(TestTileRowExpand128x128())
    #     assert result.passed, f"Test failed: {result.error}"

    # --- row_expand_add ---
    def test_tile_row_expand_add_32x128(self, test_runner):
        """Test row_expand_add with 32x128 shape."""
        result = test_runner.run(TestTileRowExpandAdd32x128())
        assert result.passed, f"Test failed: {result.error}"

    def test_tile_row_expand_add_128x128(self, test_runner):
        """Test row_expand_add with 128x128 shape."""
        result = test_runner.run(TestTileRowExpandAdd128x128())
        assert result.passed, f"Test failed: {result.error}"

    # --- row_expand_sub ---
    def test_tile_row_expand_sub_32x128(self, test_runner):
        """Test row_expand_sub with 32x128 shape."""
        result = test_runner.run(TestTileRowExpandSub32x128())
        assert result.passed, f"Test failed: {result.error}"

    def test_tile_row_expand_sub_128x128(self, test_runner):
        """Test row_expand_sub with 128x128 shape."""
        result = test_runner.run(TestTileRowExpandSub128x128())
        assert result.passed, f"Test failed: {result.error}"

    # --- row_expand_mul ---
    def test_tile_row_expand_mul_32x128(self, test_runner):
        """Test row_expand_mul with 32x128 shape."""
        result = test_runner.run(TestTileRowExpandMul32x128())
        assert result.passed, f"Test failed: {result.error}"

    def test_tile_row_expand_mul_128x128(self, test_runner):
        """Test row_expand_mul with 128x128 shape."""
        result = test_runner.run(TestTileRowExpandMul128x128())
        assert result.passed, f"Test failed: {result.error}"

    # --- row_expand_div ---
    def test_tile_row_expand_div_32x128(self, test_runner):
        """Test row_expand_div with 32x128 shape."""
        result = test_runner.run(TestTileRowExpandDiv32x128())
        assert result.passed, f"Test failed: {result.error}"

    def test_tile_row_expand_div_128x128(self, test_runner):
        """Test row_expand_div with 128x128 shape."""
        result = test_runner.run(TestTileRowExpandDiv128x128())
        assert result.passed, f"Test failed: {result.error}"

    # --- col_expand ---
    def test_tile_col_expand_32x128(self, test_runner):
        """Test col_expand with 32x128 shape."""
        result = test_runner.run(TestTileColExpand32x128())
        assert result.passed, f"Test failed: {result.error}"

    def test_tile_col_expand_128x128(self, test_runner):
        """Test col_expand with 128x128 shape."""
        result = test_runner.run(TestTileColExpand128x128())
        assert result.passed, f"Test failed: {result.error}"

    # --- col_expand_mul ---
    def test_tile_col_expand_mul_32x128(self, test_runner):
        """Test col_expand_mul with 32x128 shape."""
        result = test_runner.run(TestTileColExpandMul32x128())
        assert result.passed, f"Test failed: {result.error}"

    def test_tile_col_expand_mul_128x128(self, test_runner):
        """Test col_expand_mul with 128x128 shape."""
        result = test_runner.run(TestTileColExpandMul128x128())
        assert result.passed, f"Test failed: {result.error}"

    # --- col_expand_div ---
    def test_tile_col_expand_div_32x128(self, test_runner):
        """Test col_expand_div with 32x128 shape."""
        result = test_runner.run(TestTileColExpandDiv32x128())
        assert result.passed, f"Test failed: {result.error}"

    def test_tile_col_expand_div_128x128(self, test_runner):
        """Test col_expand_div with 128x128 shape."""
        result = test_runner.run(TestTileColExpandDiv128x128())
        assert result.passed, f"Test failed: {result.error}"

    # --- col_expand_sub ---
    def test_tile_col_expand_sub_32x128(self, test_runner):
        """Test col_expand_sub with 32x128 shape."""
        result = test_runner.run(TestTileColExpandSub32x128())
        assert result.passed, f"Test failed: {result.error}"

    def test_tile_col_expand_sub_128x128(self, test_runner):
        """Test col_expand_sub with 128x128 shape."""
        result = test_runner.run(TestTileColExpandSub128x128())
        assert result.passed, f"Test failed: {result.error}"

    # --- expands ---
    def test_tile_expands_32x128(self, test_runner):
        """Test expands (scalar to tile) with 32x128 shape."""
        result = test_runner.run(TestTileExpandScalar32x128())
        assert result.passed, f"Test failed: {result.error}"

    def test_tile_expands_128x128(self, test_runner):
        """Test expands (scalar to tile) with 128x128 shape."""
        result = test_runner.run(TestTileExpandScalar128x128())
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
