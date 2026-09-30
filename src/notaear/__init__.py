"""NotaEar: lossless ABC patches plus explainable sight-singing features."""

from .abc_patches import AbcDocument, AbcPatchError, BarPatch, split_abc
from .difficulty import DifficultyReport, analyze_abc
from .tokenizer import EncodedABC, NotaEarTokenizer, TokenizerError
from .tonality import TonalEvent, TonalityReport, analyze_tonality
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
    "TonalEvent",
    "TonalityReport",
    "ValidationIssue",
    "analyze_abc",
    "analyze_tonality",
    "split_abc",
    "validate_abc",
]
