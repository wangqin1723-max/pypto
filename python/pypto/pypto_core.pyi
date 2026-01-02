# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Type stubs for pypto_core C++ extension module."""

from typing import NoReturn

class testing:
    """Testing utilities for PyPTO error handling."""

    @staticmethod
    def raise_value_error(message: str) -> NoReturn:
        """Raise a ValueError with the given message."""
        ...

    @staticmethod
    def raise_type_error(message: str) -> NoReturn:
        """Raise a TypeError with the given message."""
        ...

    @staticmethod
    def raise_runtime_error(message: str) -> NoReturn:
        """Raise a RuntimeError with the given message."""
        ...

    @staticmethod
    def raise_not_implemented_error(message: str) -> NoReturn:
        """Raise a NotImplementedError with the given message."""
        ...

    @staticmethod
    def raise_index_error(message: str) -> NoReturn:
        """Raise an IndexError with the given message."""
        ...

    @staticmethod
    def raise_generic_error(message: str) -> NoReturn:
        """Raise a generic Error with the given message."""
        ...

__all__ = ["testing"]
