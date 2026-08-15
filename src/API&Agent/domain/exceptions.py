# DEBUG:
class ForwardOnlyException(Exception):
    pass


class NonExistentPiecePromotionException(Exception):
    pass

# VALIDATORS:
class ApiValidatorException(Exception):
    pass


class NonExistentValidMoveException(ApiValidatorException):
    pass


class NonExistentValidPieceWithUidException(ApiValidatorException):
    pass


class NonExistentBoardPositionException(ApiValidatorException):
    pass


class PlayerNotOnTurnException(ApiValidatorException):
    pass


class PiecePinnedException(ApiValidatorException):
    pass


class NotAllowedActionForEnemyPiecesException(ApiValidatorException):
    pass
