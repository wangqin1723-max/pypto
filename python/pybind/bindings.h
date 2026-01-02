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

/**
 * @file bindings.h
 * @brief Common header for all Python bindings in PyPTO
 *
 * This header declares all the binding functions for different modules.
 * Each module (error, testing, tensor, ops, etc.) should implement its
 * Bind* function and include this header.
 */

#ifndef PYTHON_PYBIND_BINDINGS_H_
#define PYTHON_PYBIND_BINDINGS_H_

#include <pybind11/pybind11.h>

namespace pypto {
namespace python {

/**
 * @brief Register error exception types and exception translator
 *
 * This function registers all PyPTO error classes with Python and sets up
 * an exception translator that converts C++ exceptions to Python exceptions
 * with full stack trace information.
 *
 * @param m The pybind11 module object
 */
void BindErrors(pybind11::module_& m);

/**
 * @brief Register testing utilities as a submodule
 *
 * Creates a protected testing submodule containing helper functions
 * for testing error handling and other internal functionality.
 *
 * @param m The parent pybind11 module object
 */
void BindTesting(pybind11::module_& m);

// Future binding declarations can be added here:
// void BindTensors(pybind11::module_& m);
// void BindOps(pybind11::module_& m);
// void BindDevices(pybind11::module_& m);

}  // namespace python
}  // namespace pypto

#endif  // PYTHON_PYBIND_BINDINGS_H_
