"""NotaEar: lossless ABC patches plus explainable sight-singing features."""

from .abc_patches import AbcDocument, AbcPatchError, BarPatch, split_abc
from .difficulty import DifficultyReport, analyze_abc
from .tokenizer import EncodedABC, NotaEarTokenizer, TokenizerError
from .validate_abc import ABCValidationReport, ValidationIssue, validate_abc

__all__ = [
    "AbcDocument",
    "AbcPatchError",
    "ABCValidationReport",
    "BarPatch",
    "DifficultyReport",
    "EncodedABC",
    "NotaEarTokenizer",
    "TokenizerError",
    "ValidationIssue",
    "analyze_abc",
    "split_abc",
    "validate_abc",
]
