class EvakuError(Exception):
    pass


class DataNotLoadedError(EvakuError):
    pass


class ConfigurationError(EvakuError):
    pass


class InvalidInputError(EvakuError):
    pass


class InvalidModelPackageError(EvakuError):
    pass
