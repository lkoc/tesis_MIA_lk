"""Exceptions shared by the case schema."""


class MissingDataError(ValueError):
    """A datum required by the physics is absent, flagged as missing, or has no provenance."""


class DimensionalError(ValueError):
    """A quantity carries a unit that does not match its physical role."""