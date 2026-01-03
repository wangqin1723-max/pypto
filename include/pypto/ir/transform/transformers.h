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

#ifndef PYPTO_IR_TRANSFORM_TRANSFORMERS_H_
#define PYPTO_IR_TRANSFORM_TRANSFORMERS_H_

#include <cstdint>

#include "pypto/ir/scalar_expr.h"

namespace pypto {
namespace ir {

/**
 * @brief Compute structural hash of an expression
 *
 * Computes hash based on expression tree structure, ignoring Span (source location).
 * Two expressions with identical structure will hash to the same value.
 *
 * This can be used with structural_equal as custom hasher/comparator for unordered containers:
 * @code
 * std::unordered_map<ExprPtr, int,
 *                    decltype(&structural_hash),
 *                    decltype(&structural_equal)> my_map(
 *   16, structural_hash, structural_equal
 * );
 * @endcode
 *
 * @param expr Expression to hash
 * @param enable_auto_mapping If true, ignore variable names (e.g., x+1 and y+1 hash the same).
 *                            If false, variable names matter (default).
 * @return Structural hash value
 */
int64_t structural_hash(const ExprPtr& expr, bool enable_auto_mapping = false);

/**
 * @brief Check if two expressions are structurally equal
 *
 * Compares expression tree structure, ignoring Span (source location).
 * Two expressions with identical structure are considered equal.
 *
 * @param lhs First expression
 * @param rhs Second expression
 * @param enable_auto_mapping If true, automatically map variables (e.g., x+1 equals y+1).
 *                            If false, variable names must match exactly (default).
 * @return true if structurally equal, false otherwise
 */
bool structural_equal(const ExprPtr& lhs, const ExprPtr& rhs, bool enable_auto_mapping = false);

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORM_TRANSFORMERS_H_
