# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Sub-module containing utilities for the Orbit framework.

* `configclass`: Provides wrapper around `dataclass` for working with configurations.
* `dict`: Provides helper functions for converting dictionaries and type-cases.
* `kit`: Provides helper functions for Omniverse kit (USD operations).
* `math`: Provides helper functions for math operations.
* `string`: Provides helper functions for string operations.
* `timer`: Provides a timer class (uses contextlib) for benchmarking.
"""

from .array import TENSOR_TYPE_CONVERSIONS, TENSOR_TYPES, TensorData, convert_to_torch
from .configclass import configclass
from .dict import class_to_dict, convert_dict_to_backend, print_dict, update_class_from_dict, update_dict
from .string import (
    callable_to_string,
    is_lambda_expression,
    resolve_matching_names,
    resolve_matching_names_values,
    string_to_callable,
    to_camel_case,
    to_snake_case,
)
from .timer import Timer

__all__ = [
    # arrays
    "TensorData",
    "TENSOR_TYPES",
    "TENSOR_TYPE_CONVERSIONS",
    "convert_to_torch",
    # config wrapper
    "configclass",
    # dictionary utilities
    "class_to_dict",
    "convert_dict_to_backend",
    "print_dict",
    "update_class_from_dict",
    "update_dict",
    # string utilities
    "to_camel_case",
    "to_snake_case",
    "is_lambda_expression",
    "string_to_callable",
    "callable_to_string",
    "resolve_matching_names",
    "resolve_matching_names_values",
    # timer
    "Timer",
]
