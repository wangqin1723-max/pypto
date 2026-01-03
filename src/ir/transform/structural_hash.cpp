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

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>

#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transform/base/functor.h"
#include "pypto/ir/transform/transformers.h"

namespace pypto {
namespace ir {

/**
 * @brief Internal implementation: Structural hash functor for expressions
 *
 * Computes hash based on expression tree structure, ignoring Span (source location).
 * This class is not part of the public API - use structural_hash() function instead.
 */
class StructuralHash : public ExprFunctor<int64_t> {
 public:
  explicit StructuralHash(bool enable_auto_mapping) : enable_auto_mapping_(enable_auto_mapping) {}
  int64_t operator()(const ExprPtr& expr);
  int64_t VisitExpr(const ExprPtr& expr) override;

 protected:
  // Leaf nodes
  int64_t VisitExpr_(const VarPtr& op) override;
  int64_t VisitExpr_(const ConstIntPtr& op) override;
  int64_t VisitExpr_(const CallPtr& op) override;

  // Binary operations
  int64_t VisitExpr_(const AddPtr& op) override;
  int64_t VisitExpr_(const SubPtr& op) override;
  int64_t VisitExpr_(const MulPtr& op) override;
  int64_t VisitExpr_(const FloorDivPtr& op) override;
  int64_t VisitExpr_(const FloorModPtr& op) override;
  int64_t VisitExpr_(const FloatDivPtr& op) override;
  int64_t VisitExpr_(const MinPtr& op) override;
  int64_t VisitExpr_(const MaxPtr& op) override;
  int64_t VisitExpr_(const PowPtr& op) override;
  int64_t VisitExpr_(const EqPtr& op) override;
  int64_t VisitExpr_(const NePtr& op) override;
  int64_t VisitExpr_(const LtPtr& op) override;
  int64_t VisitExpr_(const LePtr& op) override;
  int64_t VisitExpr_(const GtPtr& op) override;
  int64_t VisitExpr_(const GePtr& op) override;
  int64_t VisitExpr_(const AndPtr& op) override;
  int64_t VisitExpr_(const OrPtr& op) override;
  int64_t VisitExpr_(const XorPtr& op) override;
  int64_t VisitExpr_(const BitAndPtr& op) override;
  int64_t VisitExpr_(const BitOrPtr& op) override;
  int64_t VisitExpr_(const BitXorPtr& op) override;
  int64_t VisitExpr_(const BitShiftLeftPtr& op) override;
  int64_t VisitExpr_(const BitShiftRightPtr& op) override;

  // Unary operations
  int64_t VisitExpr_(const AbsPtr& op) override;
  int64_t VisitExpr_(const NegPtr& op) override;
  int64_t VisitExpr_(const NotPtr& op) override;
  int64_t VisitExpr_(const BitNotPtr& op) override;

 private:
  static int64_t hash_combine(int64_t seed, int64_t value);
  int64_t HashBinaryOp(const BinaryExprPtr& op, const char* type_name);
  int64_t HashUnaryOp(const UnaryExprPtr& op, const char* type_name);

  bool enable_auto_mapping_;
  std::unordered_map<const Var*, int64_t> var_map_;
  int64_t free_var_counter_ = 0;
};

int64_t StructuralHash::operator()(const ExprPtr& expr) { return VisitExpr(expr); }

int64_t StructuralHash::VisitExpr(const ExprPtr& expr) { return ExprFunctor::VisitExpr(expr); }

// Hash combine using Boost-inspired algorithm
int64_t StructuralHash::hash_combine(int64_t seed, int64_t value) {
  // Magic number from boost::hash_combine for good distribution
  return seed ^ (value + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

// Leaf nodes
int64_t StructuralHash::VisitExpr_(const VarPtr& op) {
  int64_t h = static_cast<int64_t>(std::hash<std::string>{}("Var"));  // Type discriminator
  if (enable_auto_mapping_) {
    auto it = var_map_.find(op.get());
    if (it != var_map_.end()) {
      h = hash_combine(h, it->second);
    } else {
      var_map_[op.get()] = free_var_counter_++;
      h = hash_combine(h, (free_var_counter_));
    }
  } else {
    h = hash_combine(h, static_cast<int64_t>(std::hash<VarPtr>{}(op)));
  }
  return h;
}

int64_t StructuralHash::VisitExpr_(const ConstIntPtr& op) {
  int64_t h = static_cast<int64_t>(std::hash<std::string>{}("ConstInt"));  // Type discriminator
  h = hash_combine(h, static_cast<int64_t>(std::hash<int>{}(op->value)));  // Field value
  return h;
}

int64_t StructuralHash::VisitExpr_(const CallPtr& op) {
  int64_t h = static_cast<int64_t>(std::hash<std::string>{}("Call"));                   // Type discriminator
  h = hash_combine(h, static_cast<int64_t>(std::hash<std::string>{}(op->op_->name_)));  // Op name

  // Hash all arguments
  for (const auto& arg : op->args_) {
    h = hash_combine(h, VisitExpr(arg));
  }

  return h;
}

// Generic binary operation hash
int64_t StructuralHash::HashBinaryOp(const BinaryExprPtr& op, const char* type_name) {
  int64_t h = static_cast<int64_t>(std::hash<std::string>{}(type_name));  // Type discriminator
  h = hash_combine(h, VisitExpr(op->left_));                              // Left child
  h = hash_combine(h, VisitExpr(op->right_));                             // Right child
  return h;
}

// Generic unary operation hash
int64_t StructuralHash::HashUnaryOp(const UnaryExprPtr& op, const char* type_name) {
  int64_t h = static_cast<int64_t>(std::hash<std::string>{}(type_name));  // Type discriminator
  h = hash_combine(h, VisitExpr(op->operand_));                           // Child
  return h;
}

// Macro to generate binary operation hash methods
#define DEFINE_BINARY_HASH(OpType) \
  int64_t StructuralHash::VisitExpr_(const OpType##Ptr& op) { return HashBinaryOp(op, #OpType); }

// Binary operations
DEFINE_BINARY_HASH(Add)
DEFINE_BINARY_HASH(Sub)
DEFINE_BINARY_HASH(Mul)
DEFINE_BINARY_HASH(FloorDiv)
DEFINE_BINARY_HASH(FloorMod)
DEFINE_BINARY_HASH(FloatDiv)
DEFINE_BINARY_HASH(Min)
DEFINE_BINARY_HASH(Max)
DEFINE_BINARY_HASH(Pow)
DEFINE_BINARY_HASH(Eq)
DEFINE_BINARY_HASH(Ne)
DEFINE_BINARY_HASH(Lt)
DEFINE_BINARY_HASH(Le)
DEFINE_BINARY_HASH(Gt)
DEFINE_BINARY_HASH(Ge)
DEFINE_BINARY_HASH(And)
DEFINE_BINARY_HASH(Or)
DEFINE_BINARY_HASH(Xor)
DEFINE_BINARY_HASH(BitAnd)
DEFINE_BINARY_HASH(BitOr)
DEFINE_BINARY_HASH(BitXor)
DEFINE_BINARY_HASH(BitShiftLeft)
DEFINE_BINARY_HASH(BitShiftRight)

#undef DEFINE_BINARY_HASH

// Macro to generate unary operation hash methods
#define DEFINE_UNARY_HASH(OpType) \
  int64_t StructuralHash::VisitExpr_(const OpType##Ptr& op) { return HashUnaryOp(op, #OpType); }

// Unary operations
DEFINE_UNARY_HASH(Abs)
DEFINE_UNARY_HASH(Neg)
DEFINE_UNARY_HASH(Not)
DEFINE_UNARY_HASH(BitNot)

#undef DEFINE_UNARY_HASH

// Public API implementation
int64_t structural_hash(const ExprPtr& expr, bool enable_auto_mapping) {
  StructuralHash hasher(enable_auto_mapping);
  return hasher(expr);
}

}  // namespace ir
}  // namespace pypto
