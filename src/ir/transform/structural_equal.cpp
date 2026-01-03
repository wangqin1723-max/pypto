/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <memory>
#include <string>
#include <unordered_map>

#include "pypto/ir/transform/transformers.h"

namespace pypto {
namespace ir {

/**
 * @brief Internal implementation: Structural equality checker for expressions
 *
 * Compares expression tree structure, ignoring Span (source location).
 * This class is not part of the public API - use structural_equal() function instead.
 */
class StructuralEqual {
 public:
  explicit StructuralEqual(bool enable_auto_mapping) : enable_auto_mapping_(enable_auto_mapping) {}
  bool operator()(const ExprPtr& lhs, const ExprPtr& rhs);

 private:
  bool Equal(const ExprPtr& lhs, const ExprPtr& rhs);
  bool EqualVar(const VarPtr& lhs, const VarPtr& rhs);
  bool EqualConstInt(const ConstIntPtr& lhs, const ConstIntPtr& rhs) const;
  bool EqualCall(const CallPtr& lhs, const CallPtr& rhs);
  bool EqualBinaryOp(const BinaryExprPtr& lhs, const BinaryExprPtr& rhs);
  bool EqualUnaryOp(const UnaryExprPtr& lhs, const UnaryExprPtr& rhs);

  bool enable_auto_mapping_;
  // Variable mapping: lhs variable pointer -> rhs variable pointer
  std::unordered_map<const Var*, const Var*> var_map_;
};

bool StructuralEqual::operator()(const ExprPtr& lhs, const ExprPtr& rhs) { return Equal(lhs, rhs); }

bool StructuralEqual::Equal(const ExprPtr& lhs, const ExprPtr& rhs) {
  // Fast path: reference equality
  if (lhs.get() == rhs.get()) return true;
  if (!lhs || !rhs) return false;

  // Type check: must be same concrete type
  if (lhs->type_name() != rhs->type_name()) return false;

  // Dispatch to type-specific handlers using dynamic_cast
  // Leaf nodes
  if (auto lhs_var = std::dynamic_pointer_cast<const Var>(lhs)) {
    return EqualVar(lhs_var, std::static_pointer_cast<const Var>(rhs));
  }
  if (auto lhs_const = std::dynamic_pointer_cast<const ConstInt>(lhs)) {
    return EqualConstInt(lhs_const, std::static_pointer_cast<const ConstInt>(rhs));
  }
  if (auto lhs_call = std::dynamic_pointer_cast<const Call>(lhs)) {
    return EqualCall(lhs_call, std::static_pointer_cast<const Call>(rhs));
  }

  // Binary operations
  if (auto lhs_add = std::dynamic_pointer_cast<const Add>(lhs)) {
    return EqualBinaryOp(lhs_add, std::static_pointer_cast<const Add>(rhs));
  }
  if (auto lhs_sub = std::dynamic_pointer_cast<const Sub>(lhs)) {
    return EqualBinaryOp(lhs_sub, std::static_pointer_cast<const Sub>(rhs));
  }
  if (auto lhs_mul = std::dynamic_pointer_cast<const Mul>(lhs)) {
    return EqualBinaryOp(lhs_mul, std::static_pointer_cast<const Mul>(rhs));
  }
  if (auto lhs_floordiv = std::dynamic_pointer_cast<const FloorDiv>(lhs)) {
    return EqualBinaryOp(lhs_floordiv, std::static_pointer_cast<const FloorDiv>(rhs));
  }
  if (auto lhs_floormod = std::dynamic_pointer_cast<const FloorMod>(lhs)) {
    return EqualBinaryOp(lhs_floormod, std::static_pointer_cast<const FloorMod>(rhs));
  }
  if (auto lhs_floatdiv = std::dynamic_pointer_cast<const FloatDiv>(lhs)) {
    return EqualBinaryOp(lhs_floatdiv, std::static_pointer_cast<const FloatDiv>(rhs));
  }
  if (auto lhs_min = std::dynamic_pointer_cast<const Min>(lhs)) {
    return EqualBinaryOp(lhs_min, std::static_pointer_cast<const Min>(rhs));
  }
  if (auto lhs_max = std::dynamic_pointer_cast<const Max>(lhs)) {
    return EqualBinaryOp(lhs_max, std::static_pointer_cast<const Max>(rhs));
  }
  if (auto lhs_pow = std::dynamic_pointer_cast<const Pow>(lhs)) {
    return EqualBinaryOp(lhs_pow, std::static_pointer_cast<const Pow>(rhs));
  }
  if (auto lhs_eq = std::dynamic_pointer_cast<const Eq>(lhs)) {
    return EqualBinaryOp(lhs_eq, std::static_pointer_cast<const Eq>(rhs));
  }
  if (auto lhs_ne = std::dynamic_pointer_cast<const Ne>(lhs)) {
    return EqualBinaryOp(lhs_ne, std::static_pointer_cast<const Ne>(rhs));
  }
  if (auto lhs_lt = std::dynamic_pointer_cast<const Lt>(lhs)) {
    return EqualBinaryOp(lhs_lt, std::static_pointer_cast<const Lt>(rhs));
  }
  if (auto lhs_le = std::dynamic_pointer_cast<const Le>(lhs)) {
    return EqualBinaryOp(lhs_le, std::static_pointer_cast<const Le>(rhs));
  }
  if (auto lhs_gt = std::dynamic_pointer_cast<const Gt>(lhs)) {
    return EqualBinaryOp(lhs_gt, std::static_pointer_cast<const Gt>(rhs));
  }
  if (auto lhs_ge = std::dynamic_pointer_cast<const Ge>(lhs)) {
    return EqualBinaryOp(lhs_ge, std::static_pointer_cast<const Ge>(rhs));
  }
  if (auto lhs_and = std::dynamic_pointer_cast<const And>(lhs)) {
    return EqualBinaryOp(lhs_and, std::static_pointer_cast<const And>(rhs));
  }
  if (auto lhs_or = std::dynamic_pointer_cast<const Or>(lhs)) {
    return EqualBinaryOp(lhs_or, std::static_pointer_cast<const Or>(rhs));
  }
  if (auto lhs_xor = std::dynamic_pointer_cast<const Xor>(lhs)) {
    return EqualBinaryOp(lhs_xor, std::static_pointer_cast<const Xor>(rhs));
  }
  if (auto lhs_bitand = std::dynamic_pointer_cast<const BitAnd>(lhs)) {
    return EqualBinaryOp(lhs_bitand, std::static_pointer_cast<const BitAnd>(rhs));
  }
  if (auto lhs_bitor = std::dynamic_pointer_cast<const BitOr>(lhs)) {
    return EqualBinaryOp(lhs_bitor, std::static_pointer_cast<const BitOr>(rhs));
  }
  if (auto lhs_bitxor = std::dynamic_pointer_cast<const BitXor>(lhs)) {
    return EqualBinaryOp(lhs_bitxor, std::static_pointer_cast<const BitXor>(rhs));
  }
  if (auto lhs_bitshiftleft = std::dynamic_pointer_cast<const BitShiftLeft>(lhs)) {
    return EqualBinaryOp(lhs_bitshiftleft, std::static_pointer_cast<const BitShiftLeft>(rhs));
  }
  if (auto lhs_bitshiftright = std::dynamic_pointer_cast<const BitShiftRight>(lhs)) {
    return EqualBinaryOp(lhs_bitshiftright, std::static_pointer_cast<const BitShiftRight>(rhs));
  }

  // Unary operations
  if (auto lhs_abs = std::dynamic_pointer_cast<const Abs>(lhs)) {
    return EqualUnaryOp(lhs_abs, std::static_pointer_cast<const Abs>(rhs));
  }
  if (auto lhs_neg = std::dynamic_pointer_cast<const Neg>(lhs)) {
    return EqualUnaryOp(lhs_neg, std::static_pointer_cast<const Neg>(rhs));
  }
  if (auto lhs_not = std::dynamic_pointer_cast<const Not>(lhs)) {
    return EqualUnaryOp(lhs_not, std::static_pointer_cast<const Not>(rhs));
  }
  if (auto lhs_bitnot = std::dynamic_pointer_cast<const BitNot>(lhs)) {
    return EqualUnaryOp(lhs_bitnot, std::static_pointer_cast<const BitNot>(rhs));
  }

  // Unknown type
  return false;
}

bool StructuralEqual::EqualVar(const VarPtr& lhs, const VarPtr& rhs) {
  if (!enable_auto_mapping_) {
    // Without auto mapping, require exact pointer match (strict identity)
    return lhs.get() == rhs.get();
  }

  // With auto mapping: maintain consistent variable mapping using pointers
  // This allows x+1 to equal y+1 by mapping x->y
  auto it = var_map_.find(lhs.get());
  if (it != var_map_.end()) {
    // Variable already mapped, verify consistency (same pointer)
    return it->second == rhs.get();
  }

  // New variable, add to mapping
  var_map_[lhs.get()] = rhs.get();
  return true;
}

bool StructuralEqual::EqualConstInt(const ConstIntPtr& lhs, const ConstIntPtr& rhs) const {
  // Compare constant value (ignore span)
  return lhs->value == rhs->value;
}

bool StructuralEqual::EqualCall(const CallPtr& lhs, const CallPtr& rhs) {
  // Compare op name
  if (lhs->op_->name_ != rhs->op_->name_) return false;

  // Compare argument count
  if (lhs->args_.size() != rhs->args_.size()) return false;

  // Recursively compare all arguments
  for (size_t i = 0; i < lhs->args_.size(); ++i) {
    if (!Equal(lhs->args_[i], rhs->args_[i])) return false;
  }

  return true;
}

bool StructuralEqual::EqualBinaryOp(const BinaryExprPtr& lhs, const BinaryExprPtr& rhs) {
  // Recursively compare left and right children
  return Equal(lhs->left_, rhs->left_) && Equal(lhs->right_, rhs->right_);
}

bool StructuralEqual::EqualUnaryOp(const UnaryExprPtr& lhs, const UnaryExprPtr& rhs) {
  // Recursively compare operand
  return Equal(lhs->operand_, rhs->operand_);
}

// Public API implementation
bool structural_equal(const ExprPtr& lhs, const ExprPtr& rhs, bool enable_auto_mapping) {
  StructuralEqual checker(enable_auto_mapping);
  return checker(lhs, rhs);
}

}  // namespace ir
}  // namespace pypto
