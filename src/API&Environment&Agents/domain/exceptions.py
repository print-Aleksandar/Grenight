class GrenightException(Exception):
    pass


class NonExistentPiecePromotionException(GrenightException):
    pass


class NonExistentValidMoveException(GrenightException):
    pass


class PiecePinnedException(GrenightException):
    pass


class NonExistentValidPieceWithUidException(GrenightException):
    pass


class NonExistentBoardPositionException(GrenightException):
    pass


class PlayerNotOnMoveTurnException(GrenightException):
    pass


class PlayerNotOnPieceValidMovesTurnException(GrenightException):
    pass


class PlayerWantsToPlayWithEnemyPieceException(GrenightException):
    pass


class PlayerWantsToGatherValidMovesForEnemyPieceException(GrenightException):
    pass


class AgentNotOnTurnException(GrenightException):
    pass
