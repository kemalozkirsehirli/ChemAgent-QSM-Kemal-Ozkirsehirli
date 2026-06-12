class ChemAgentQSMError(RuntimeError):
    """Base exception for ChemAgent-QSM."""


class ValidationError(ChemAgentQSMError):
    """Raised when a workflow plan fails static validation."""


class OptionalDependencyError(ChemAgentQSMError):
    """Raised when an optional scientific backend is unavailable."""
