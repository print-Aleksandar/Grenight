class NonExistentPiecePromotionException(Exception):
    pass


class NonExistentValidMoveException(Exception):
    pass


class PiecePinnedException(Exception):
    pass


class TryingToTakeEnemyKingException(Exception):
    pass


# VALIDATORS:
class ApiValidatorException(Exception):
    pass


class NonExistentValidPieceWithUidException(ApiValidatorException):
    pass


class NonExistentBoardPositionException(ApiValidatorException):
    pass


class PlayerNotOnTurnException(ApiValidatorException):
    pass


class NotAllowedActionForEnemyPiecesException(ApiValidatorException):
    pass
