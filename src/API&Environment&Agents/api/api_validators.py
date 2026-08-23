from domain.requests import MoveRequest, ValidMovesPieceRequest, AgentMoveRequest
from domain.board_initialization import POSITIONS
from application.board_getter import get_piece_by_uid
from domain.exceptions import (NonExistentValidPieceWithUidException,
                               NonExistentBoardPositionException,
                               PlayerNotOnMoveTurnException,
                               PlayerNotOnPieceValidMovesTurnException,
                               PlayerWantsToPlayWithEnemyPieceException,
                               PlayerWantsToGatherValidMovesForEnemyPieceException,
                               AgentNotOnTurnException)


# MOVE:
def non_existent_valid_piece_with_uid_exception(request: MoveRequest) -> None:

    if request.uid not in [piece.uid for piece in request.pieces]:
        raise NonExistentValidPieceWithUidException()


def non_existent_board_position_exception(request: MoveRequest) -> None:

    if request.position not in POSITIONS:
        raise NonExistentBoardPositionException()


def player_not_on_turn_move_exception(request: MoveRequest) -> None:

    if request.is_white_on_turn != request.is_from_white_player:
        raise PlayerNotOnMoveTurnException()


def player_wants_to_play_with_enemy_pieces_exception(request: MoveRequest) -> None:

    if get_piece_by_uid(request.pieces, request.uid).is_white != request.is_from_white_player:
        raise PlayerWantsToPlayWithEnemyPieceException()


# PIECE VALID MOVES:
def player_not_on_piece_valid_moves_turn_exception(request: ValidMovesPieceRequest) -> None:

    if request.is_white_on_turn != request.is_from_white_player:
        raise PlayerNotOnPieceValidMovesTurnException()

def player_wants_to_gather_valid_moves_for_enemy_piece_exception(request: ValidMovesPieceRequest) -> None:

    if get_piece_by_uid(request.pieces, request.uid).is_white != request.is_from_white_player:
        raise PlayerWantsToGatherValidMovesForEnemyPieceException()

# AGENT:
def agent_not_on_turn_exception(request: AgentMoveRequest) -> None:

    if request.is_for_white != request.is_for_white_turn:
        raise AgentNotOnTurnException()
