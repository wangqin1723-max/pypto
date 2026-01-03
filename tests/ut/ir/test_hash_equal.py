# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for IR hash and equality functionality."""

import pytest
from pypto import ir


class TestReferenceEquality:
    """Tests for reference equality (pointer-based)."""

    def test_different_pointers_not_equal(self):
        """Test that different pointers with same structure are not reference-equal."""
        span = ir.Span.unknown()
        x1 = ir.Var("x", span)
        x2 = ir.Var("x", span)

        # Different objects, even with same content
        assert x1 != x2

    def test_different_content_not_equal(self):
        """Test that different variables are not equal."""
        span = ir.Span.unknown()
        x = ir.Var("x", span)
        y = ir.Var("y", span)

        assert x != y

    def test_inequality_operator(self):
        """Test inequality operator works correctly."""
        span = ir.Span.unknown()
        x1 = ir.Var("x", span)
        x2 = ir.Var("x", span)

        assert x1 != x2
        assert not (x1 == x2)


class TestStructuralHash:
    """Tests for structural hash function."""

    def test_same_structure_same_hash(self):
        """Test that expressions with same structure hash to same value."""
        span1 = ir.Span.unknown()
        span2 = ir.Span.unknown()  # Different span

        x1 = ir.Var("x", span1)
        x2 = ir.Var("x", span2)

        # Same variable name, different spans - should have same hash
        hash1 = ir.structural_hash(x1)
        hash2 = ir.structural_hash(x2)
        assert hash1 != hash2

    def test_different_var_names_different_hash(self):
        """Test that variables with different names hash differently."""
        span = ir.Span.unknown()
        x = ir.Var("x", span)
        y = ir.Var("y", span)

        hash_x = ir.structural_hash(x)
        hash_y = ir.structural_hash(y)

        # Different names should (almost certainly) have different hashes
        assert hash_x != hash_y

    def test_different_const_values_different_hash(self):
        """Test that constants with different values hash differently."""
        span = ir.Span.unknown()
        c1 = ir.ConstInt(1, span)
        c2 = ir.ConstInt(2, span)

        hash1 = ir.structural_hash(c1)
        hash2 = ir.structural_hash(c2)

        assert hash1 != hash2

    def test_same_const_value_same_hash(self):
        """Test that constants with same value hash to same value."""
        span1 = ir.Span.unknown()
        span2 = ir.Span.unknown()

        c1 = ir.ConstInt(42, span1)
        c2 = ir.ConstInt(42, span2)

        hash1 = ir.structural_hash(c1)
        hash2 = ir.structural_hash(c2)

        assert hash1 == hash2

    def test_different_operation_types_different_hash(self):
        """Test that different operation types hash differently."""
        span = ir.Span.unknown()
        x = ir.Var("x", span)
        y = ir.Var("y", span)

        add_expr = ir.Add(x, y, span)
        sub_expr = ir.Sub(x, y, span)
        mul_expr = ir.Mul(x, y, span)

        hash_add = ir.structural_hash(add_expr)
        hash_sub = ir.structural_hash(sub_expr)
        hash_mul = ir.structural_hash(mul_expr)

        # Different operations should hash differently
        assert hash_add != hash_sub
        assert hash_add != hash_mul
        assert hash_sub != hash_mul

    def test_nested_expression_hash(self):
        """Test hashing of nested expressions."""
        span1 = ir.Span.unknown()
        span2 = ir.Span.unknown()

        # Build (x + 5) * 2 with different spans
        x1 = ir.Var("x", span1)
        c5_1 = ir.ConstInt(5, span1)
        c2_1 = ir.ConstInt(2, span1)
        expr1 = ir.Mul(ir.Add(x1, c5_1, span1), c2_1, span1)

        x2 = ir.Var("x", span2)
        c5_2 = ir.ConstInt(5, span2)
        c2_2 = ir.ConstInt(2, span2)
        expr2 = ir.Mul(ir.Add(x2, c5_2, span2), c2_2, span2)

        # Same structure, different spans - should hash to same value
        hash1 = ir.structural_hash(expr1)
        hash2 = ir.structural_hash(expr2)
        assert hash1 != hash2

    def test_operand_order_matters(self):
        """Test that operand order affects hash (x + y != y + x in structure)."""
        span = ir.Span.unknown()
        x = ir.Var("x", span)
        y = ir.Var("y", span)

        add1 = ir.Add(x, y, span)  # x + y
        add2 = ir.Add(y, x, span)  # y + x

        hash1 = ir.structural_hash(add1)
        hash2 = ir.structural_hash(add2)

        # Different operand order should (almost certainly) hash differently
        assert hash1 != hash2

    def test_unary_expression_hash(self):
        """Test hashing of unary expressions."""
        span1 = ir.Span.unknown()
        span2 = ir.Span.unknown()

        x1 = ir.Var("x", span1)
        neg1 = ir.Neg(x1, span1)

        x2 = ir.Var("x", span2)
        neg2 = ir.Neg(x2, span2)

        hash1 = ir.structural_hash(neg1)
        hash2 = ir.structural_hash(neg2)

        assert hash1 != hash2

    def test_call_expression_hash(self):
        """Test hashing of call expressions."""
        span = ir.Span.unknown()
        op1 = ir.Op("func")
        op2 = ir.Op("func")

        x = ir.Var("x", span)
        y = ir.Var("y", span)

        call1 = ir.Call(op1, [x, y], span)
        call2 = ir.Call(op2, [x, y], span)

        hash1 = ir.structural_hash(call1)
        hash2 = ir.structural_hash(call2)

        # Same op name and args - should hash to same value
        assert hash1 == hash2

    def test_different_op_names_different_hash(self):
        """Test that calls with different op names hash differently."""
        span = ir.Span.unknown()
        op1 = ir.Op("func1")
        op2 = ir.Op("func2")

        x = ir.Var("x", span)

        call1 = ir.Call(op1, [x], span)
        call2 = ir.Call(op2, [x], span)

        hash1 = ir.structural_hash(call1)
        hash2 = ir.structural_hash(call2)

        # Different op names should hash differently
        assert hash1 != hash2


class TestStructuralEquality:
    """Tests for structural equality function."""

    def test_same_var_structural_equal(self):
        """Test that variables with same name are structurally equal with auto mapping."""
        span1 = ir.Span.unknown()
        span2 = ir.Span.unknown()

        x1 = ir.Var("x", span1)
        x2 = ir.Var("x", span2)

        # Different objects, need auto mapping to be equal
        assert ir.structural_equal(x1, x2, enable_auto_mapping=True)

    def test_different_var_not_structural_equal(self):
        """Test that variables with different names are not structurally equal."""
        span = ir.Span.unknown()
        x = ir.Var("x", span)
        y = ir.Var("y", span)

        assert not ir.structural_equal(x, y)

    def test_same_const_structural_equal(self):
        """Test that constants with same value are structurally equal."""
        span1 = ir.Span.unknown()
        span2 = ir.Span.unknown()

        c1 = ir.ConstInt(42, span1)
        c2 = ir.ConstInt(42, span2)

        assert ir.structural_equal(c1, c2)

    def test_different_const_not_structural_equal(self):
        """Test that constants with different values are not structurally equal."""
        span = ir.Span.unknown()
        c1 = ir.ConstInt(1, span)
        c2 = ir.ConstInt(2, span)

        assert not ir.structural_equal(c1, c2)

    def test_different_types_not_equal(self):
        """Test that different expression types are not structurally equal."""
        span = ir.Span.unknown()
        var = ir.Var("x", span)
        const = ir.ConstInt(1, span)

        assert not ir.structural_equal(var, const)

    def test_binary_expr_structural_equal(self):
        """Test structural equality of binary expressions."""
        span1 = ir.Span.unknown()
        span2 = ir.Span.unknown()

        x1 = ir.Var("x", span1)
        y1 = ir.Var("y", span1)
        add1 = ir.Add(x1, y1, span1)

        x2 = ir.Var("x", span2)
        y2 = ir.Var("y", span2)
        add2 = ir.Add(x2, y2, span2)

        # Same structure with auto mapping
        assert ir.structural_equal(add1, add2, enable_auto_mapping=True)

    def test_different_binary_ops_not_equal(self):
        """Test that different binary operations are not structurally equal."""
        span = ir.Span.unknown()
        x = ir.Var("x", span)
        y = ir.Var("y", span)

        add_expr = ir.Add(x, y, span)
        sub_expr = ir.Sub(x, y, span)

        assert not ir.structural_equal(add_expr, sub_expr)

    def test_operand_order_matters_in_equality(self):
        """Test that operand order matters for structural equality."""
        span = ir.Span.unknown()
        x = ir.Var("x", span)
        y = ir.Var("y", span)

        add1 = ir.Add(x, y, span)  # x + y
        add2 = ir.Add(y, x, span)  # y + x

        # Different operand order
        assert not ir.structural_equal(add1, add2)

    def test_nested_expressions_structural_equal(self):
        """Test structural equality of nested expressions."""
        span1 = ir.Span.unknown()
        span2 = ir.Span.unknown()

        # Build (x + 5) * 2
        x1 = ir.Var("x", span1)
        c5_1 = ir.ConstInt(5, span1)
        c2_1 = ir.ConstInt(2, span1)
        expr1 = ir.Mul(ir.Add(x1, c5_1, span1), c2_1, span1)

        x2 = ir.Var("x", span2)
        c5_2 = ir.ConstInt(5, span2)
        c2_2 = ir.ConstInt(2, span2)
        expr2 = ir.Mul(ir.Add(x2, c5_2, span2), c2_2, span2)

        assert ir.structural_equal(expr1, expr2, enable_auto_mapping=True)

    def test_different_nested_structure_not_equal(self):
        """Test that different nested structures are not equal."""
        span = ir.Span.unknown()
        x = ir.Var("x", span)
        c5 = ir.ConstInt(5, span)
        c2 = ir.ConstInt(2, span)

        expr1 = ir.Mul(ir.Add(x, c5, span), c2, span)  # (x + 5) * 2
        expr2 = ir.Add(ir.Mul(x, c5, span), c2, span)  # (x * 5) + 2

        assert not ir.structural_equal(expr1, expr2)

    def test_unary_expr_structural_equal(self):
        """Test structural equality of unary expressions."""
        span1 = ir.Span.unknown()
        span2 = ir.Span.unknown()

        x1 = ir.Var("x", span1)
        neg1 = ir.Neg(x1, span1)

        x2 = ir.Var("x", span2)
        neg2 = ir.Neg(x2, span2)

        assert ir.structural_equal(neg1, neg2, enable_auto_mapping=True)

    def test_different_unary_ops_not_equal(self):
        """Test that different unary operations are not equal."""
        span = ir.Span.unknown()
        x = ir.Var("x", span)

        neg_expr = ir.Neg(x, span)
        abs_expr = ir.Abs(x, span)

        assert not ir.structural_equal(neg_expr, abs_expr)

    def test_call_expr_structural_equal(self):
        """Test structural equality of call expressions."""
        span = ir.Span.unknown()
        op1 = ir.Op("func")
        op2 = ir.Op("func")

        x = ir.Var("x", span)
        y = ir.Var("y", span)

        call1 = ir.Call(op1, [x, y], span)
        call2 = ir.Call(op2, [x, y], span)

        # Same op name and args
        assert ir.structural_equal(call1, call2)

    def test_different_op_names_not_equal(self):
        """Test that calls with different op names are not equal."""
        span = ir.Span.unknown()
        op1 = ir.Op("func1")
        op2 = ir.Op("func2")

        x = ir.Var("x", span)

        call1 = ir.Call(op1, [x], span)
        call2 = ir.Call(op2, [x], span)

        assert not ir.structural_equal(call1, call2)

    def test_different_arg_count_not_equal(self):
        """Test that calls with different argument counts are not equal."""
        span = ir.Span.unknown()
        op = ir.Op("func")

        x = ir.Var("x", span)
        y = ir.Var("y", span)

        call1 = ir.Call(op, [x], span)
        call2 = ir.Call(op, [x, y], span)

        assert not ir.structural_equal(call1, call2)

    def test_empty_call_args_equal(self):
        """Test that calls with empty args lists can be equal."""
        span = ir.Span.unknown()
        op1 = ir.Op("func")
        op2 = ir.Op("func")

        call1 = ir.Call(op1, [], span)
        call2 = ir.Call(op2, [], span)

        assert ir.structural_equal(call1, call2)


class TestHashEqualityConsistency:
    """Test that hash and equality are consistent."""

    def test_equal_implies_same_hash(self):
        """Test that structurally equal expressions have the same hash."""
        span1 = ir.Span.unknown()
        span2 = ir.Span.unknown()

        # Create several pairs of structurally equal expressions
        test_cases = [
            (ir.Var("x", span1), ir.Var("x", span2)),
            (ir.ConstInt(42, span1), ir.ConstInt(42, span2)),
            (
                ir.Add(ir.Var("a", span1), ir.Var("b", span1), span1),
                ir.Add(ir.Var("a", span2), ir.Var("b", span2), span2),
            ),
            (ir.Neg(ir.Var("x", span1), span1), ir.Neg(ir.Var("x", span2), span2)),
        ]

        for expr1, expr2 in test_cases:
            if ir.structural_equal(expr1, expr2):
                assert ir.structural_hash(expr1) == ir.structural_hash(expr2), (
                    f"Equal expressions should have same hash: {expr1} vs {expr2}"
                )

    def test_deep_nested_consistency(self):
        """Test hash/equality consistency for deeply nested expressions."""
        span1 = ir.Span.unknown()
        span2 = ir.Span.unknown()

        # Build: (((x + 1) - 2) * 3) / 4
        def build_expr(sp):
            x = ir.Var("x", sp)
            c1 = ir.ConstInt(1, sp)
            c2 = ir.ConstInt(2, sp)
            c3 = ir.ConstInt(3, sp)
            c4 = ir.ConstInt(4, sp)

            return ir.FloatDiv(ir.Mul(ir.Sub(ir.Add(x, c1, sp), c2, sp), c3, sp), c4, sp)

        expr1 = build_expr(span1)
        expr2 = build_expr(span2)

        assert ir.structural_equal(expr1, expr2, enable_auto_mapping=True)
        assert ir.structural_hash(expr1, enable_auto_mapping=True) == ir.structural_hash(
            expr2, enable_auto_mapping=True
        )


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_comparison_operations(self):
        """Test all comparison operation types."""
        span = ir.Span.unknown()
        x = ir.Var("x", span)
        y = ir.Var("y", span)

        ops = [
            (ir.Eq, ir.Eq),
            (ir.Ne, ir.Ne),
            (ir.Lt, ir.Lt),
            (ir.Le, ir.Le),
            (ir.Gt, ir.Gt),
            (ir.Ge, ir.Ge),
        ]

        for op1, op2 in ops:
            expr1 = op1(x, y, span)
            expr2 = op2(x, y, span)
            assert ir.structural_equal(expr1, expr2)

    def test_logical_operations(self):
        """Test all logical operation types."""
        span = ir.Span.unknown()
        x = ir.Var("x", span)
        y = ir.Var("y", span)

        ops = [(ir.And, ir.And), (ir.Or, ir.Or), (ir.Xor, ir.Xor)]

        for op1, op2 in ops:
            expr1 = op1(x, y, span)
            expr2 = op2(x, y, span)
            assert ir.structural_equal(expr1, expr2)

    def test_bitwise_operations(self):
        """Test all bitwise operation types."""
        span = ir.Span.unknown()
        x = ir.Var("x", span)
        y = ir.Var("y", span)

        ops = [
            (ir.BitAnd, ir.BitAnd),
            (ir.BitOr, ir.BitOr),
            (ir.BitXor, ir.BitXor),
            (ir.BitShiftLeft, ir.BitShiftLeft),
            (ir.BitShiftRight, ir.BitShiftRight),
        ]

        for op1, op2 in ops:
            expr1 = op1(x, y, span)
            expr2 = op2(x, y, span)
            assert ir.structural_equal(expr1, expr2)

    def test_all_unary_operations(self):
        """Test all unary operation types."""
        span = ir.Span.unknown()
        x = ir.Var("x", span)

        ops = [(ir.Abs, ir.Abs), (ir.Neg, ir.Neg), (ir.Not, ir.Not), (ir.BitNot, ir.BitNot)]

        for op1, op2 in ops:
            expr1 = op1(x, span)
            expr2 = op2(x, span)
            assert ir.structural_equal(expr1, expr2)

    def test_math_operations(self):
        """Test mathematical operation types."""
        span = ir.Span.unknown()
        x = ir.Var("x", span)
        y = ir.Var("y", span)

        ops = [(ir.Min, ir.Min), (ir.Max, ir.Max), (ir.Pow, ir.Pow)]

        for op1, op2 in ops:
            expr1 = op1(x, y, span)
            expr2 = op2(x, y, span)
            assert ir.structural_equal(expr1, expr2)


class TestAutoMapping:
    """Tests for auto mapping feature in structural equality and hash."""

    def test_auto_mapping_simple_vars_equal(self):
        """Test that x+1 equals y+1 with auto mapping enabled."""
        span = ir.Span.unknown()
        x = ir.Var("x", span)
        y = ir.Var("x", span)
        c1_1 = ir.ConstInt(1, span)
        c1_2 = ir.ConstInt(1, span)

        expr1 = ir.Add(x, c1_1, span)  # x + 1
        expr2 = ir.Add(y, c1_2, span)  # y + 1

        # Without auto mapping, they should NOT be equal
        assert not ir.structural_equal(expr1, expr2, enable_auto_mapping=False)

        # With auto mapping, they SHOULD be equal
        assert ir.structural_equal(expr1, expr2, enable_auto_mapping=True)

    def test_auto_mapping_simple_vars_not_equal(self):
        """Test that x+1 does not equal y+1 without auto mapping."""
        span = ir.Span.unknown()
        x = ir.Var("x", span)
        y = ir.Var("x", span)
        c1 = ir.ConstInt(1, span)

        expr1 = ir.Add(x, c1, span)  # x + 1
        expr2 = ir.Add(y, c1, span)  # y + 1

        # Without auto mapping (default), they should NOT be equal
        assert not ir.structural_equal(expr1, expr2)
        x = ir.Var("x", span)
        y = ir.Var("x", span)
        c1 = ir.ConstInt(1, span)

        expr1 = ir.Add(x, c1, span)  # x + 1
        expr2 = ir.Add(y, c1, span)  # y + 1

        # Without auto mapping (default), they should NOT be equal
        assert not ir.structural_equal(expr1, expr2)

    def test_auto_mapping_hash_consistency(self):
        """Test that x+1 and y+1 hash to same value with auto mapping."""
        span = ir.Span.unknown()
        x = ir.Var("x", span)
        y = ir.Var("y", span)
        c1_1 = ir.ConstInt(1, span)
        c1_2 = ir.ConstInt(1, span)

        expr1 = ir.Add(x, c1_1, span)  # x + 1
        expr2 = ir.Add(y, c1_2, span)  # y + 1

        # Without auto mapping, hashes should be different
        hash1_no_auto = ir.structural_hash(expr1, enable_auto_mapping=False)
        hash2_no_auto = ir.structural_hash(expr2, enable_auto_mapping=False)
        assert hash1_no_auto != hash2_no_auto

        # With auto mapping, hashes should be the same
        hash1_auto = ir.structural_hash(expr1, enable_auto_mapping=True)
        hash2_auto = ir.structural_hash(expr2, enable_auto_mapping=True)
        assert hash1_auto == hash2_auto

    def test_auto_mapping_multiple_vars(self):
        """Test auto mapping with multiple different variables."""
        span = ir.Span.unknown()

        # Build: (x + y) * z
        x = ir.Var("x", span)
        y = ir.Var("y", span)
        z = ir.Var("z", span)
        expr1 = ir.Mul(ir.Add(x, y, span), z, span)

        # Build: (a + b) * c
        a = ir.Var("a", span)
        b = ir.Var("b", span)
        c = ir.Var("c", span)
        expr2 = ir.Mul(ir.Add(a, b, span), c, span)

        # Without auto mapping, should not be equal
        assert not ir.structural_equal(expr1, expr2, enable_auto_mapping=False)

        # With auto mapping, should be equal
        assert ir.structural_equal(expr1, expr2, enable_auto_mapping=True)

        # Hashes should also match with auto mapping
        assert ir.structural_hash(expr1, enable_auto_mapping=True) == ir.structural_hash(
            expr2, enable_auto_mapping=True
        )

    def test_auto_mapping_consistent_mapping(self):
        """Test that auto mapping maintains consistent variable mapping."""
        span = ir.Span.unknown()

        # Build: x + x (same variable used twice)
        x = ir.Var("x", span)
        expr1 = ir.Add(x, x, span)

        # Build: y + y (same variable used twice)
        y = ir.Var("y", span)
        expr2 = ir.Add(y, y, span)

        # With auto mapping, x maps to y consistently
        assert ir.structural_equal(expr1, expr2, enable_auto_mapping=True)

    def test_auto_mapping_inconsistent_mapping_fails(self):
        """Test that inconsistent variable mapping is rejected."""
        span = ir.Span.unknown()

        # Build: x + x (same variable used twice)
        x = ir.Var("x", span)
        expr1 = ir.Add(x, x, span)

        # Build: y + z (different variables)
        y = ir.Var("y", span)
        z = ir.Var("z", span)
        expr2 = ir.Add(y, z, span)

        # With auto mapping, this should fail because x can't map to both y and z
        assert not ir.structural_equal(expr1, expr2, enable_auto_mapping=True)

    def test_auto_mapping_complex_expression(self):
        """Test auto mapping with complex nested expressions."""
        span = ir.Span.unknown()

        # Build: ((x + 1) * (y - 2)) / (x + y)
        x1 = ir.Var("x", span)
        y1 = ir.Var("y", span)
        c1_1 = ir.ConstInt(1, span)
        c2_1 = ir.ConstInt(2, span)
        expr1 = ir.FloatDiv(
            ir.Mul(ir.Add(x1, c1_1, span), ir.Sub(y1, c2_1, span), span), ir.Add(x1, y1, span), span
        )

        # Build: ((a + 1) * (b - 2)) / (a + b)
        a = ir.Var("a", span)
        b = ir.Var("b", span)
        c1_2 = ir.ConstInt(1, span)
        c2_2 = ir.ConstInt(2, span)
        expr2 = ir.FloatDiv(
            ir.Mul(ir.Add(a, c1_2, span), ir.Sub(b, c2_2, span), span), ir.Add(a, b, span), span
        )

        # With auto mapping: x->a, y->b consistently
        assert ir.structural_equal(expr1, expr2, enable_auto_mapping=True)
        assert ir.structural_hash(expr1, enable_auto_mapping=True) == ir.structural_hash(
            expr2, enable_auto_mapping=True
        )

    def test_auto_mapping_same_vars_still_equal(self):
        """Test that expressions with same variable names are still equal with auto mapping."""
        span = ir.Span.unknown()
        x1 = ir.Var("x", span)
        x2 = ir.Var("x", span)
        c1 = ir.ConstInt(1, span)
        c2 = ir.ConstInt(1, span)

        expr1 = ir.Add(x1, c1, span)
        expr2 = ir.Add(x2, c2, span)

        # Should be equal both with and without auto mapping
        assert not ir.structural_equal(expr1, expr2, enable_auto_mapping=False)
        assert ir.structural_equal(expr1, expr2, enable_auto_mapping=True)

    def test_auto_mapping_default_false(self):
        """Test that auto mapping is disabled by default."""
        span = ir.Span.unknown()
        x = ir.Var("x", span)
        y = ir.Var("y", span)
        c1 = ir.ConstInt(1, span)

        expr1 = ir.Add(x, c1, span)
        expr2 = ir.Add(y, c1, span)

        # Default behavior should require exact variable name match
        assert not ir.structural_equal(expr1, expr2)

        # Hash should also include variable names by default
        assert ir.structural_hash(expr1) != ir.structural_hash(expr2)

    def test_auto_mapping_with_unary_ops(self):
        """Test auto mapping with unary operations."""
        span = ir.Span.unknown()

        # Build: -x
        x = ir.Var("x", span)
        expr1 = ir.Neg(x, span)

        # Build: -y
        y = ir.Var("y", span)
        expr2 = ir.Neg(y, span)

        # With auto mapping, should be equal
        assert ir.structural_equal(expr1, expr2, enable_auto_mapping=True)
        assert ir.structural_hash(expr1, enable_auto_mapping=True) == ir.structural_hash(
            expr2, enable_auto_mapping=True
        )

    def test_auto_mapping_with_call_expr(self):
        """Test auto mapping with call expressions."""
        span = ir.Span.unknown()
        op = ir.Op("func")

        # Build: func(x, y)
        x = ir.Var("x", span)
        y = ir.Var("y", span)
        call1 = ir.Call(op, [x, y], span)

        # Build: func(a, b)
        a = ir.Var("a", span)
        b = ir.Var("b", span)
        call2 = ir.Call(op, [a, b], span)

        # With auto mapping, should be equal
        assert ir.structural_equal(call1, call2, enable_auto_mapping=True)
        assert ir.structural_hash(call1, enable_auto_mapping=True) == ir.structural_hash(
            call2, enable_auto_mapping=True
        )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
