from domain.requests import MoveRequest, ValidMovesPlayerRequest, ValidMovesPieceRequest
from domain.board_configs import POSITIONS
from application.board_getter import get_piece_by_uid
from domain.exceptions import (NonExistentValidPieceWithUidException,
                               NonExistentBoardPositionException,
                               PlayerNotOnTurnException,
                               NotAllowedActionForEnemyPiecesException)


# MOVE:
def non_existent_valid_piece_with_uid_exception(request: MoveRequest) -> None:

    if request.uid not in [piece.uid for piece in request.pieces]:
        raise NonExistentValidPieceWithUidException()


def non_existent_board_position_exception(request: MoveRequest) -> None:

    if request.position not in POSITIONS:
        raise NonExistentBoardPositionException()


def player_not_on_turn_exception(request: MoveRequest) -> None:

    if request.is_white_on_turn != request.is_from_white_player:
        raise PlayerNotOnTurnException()


def player_playing_with_enemy_pieces_exception(request: MoveRequest) -> None:

    if get_piece_by_uid(request.pieces, request.uid).is_white != request.is_from_white_player:
        raise NotAllowedActionForEnemyPiecesException()


# PLAYER_VALID_MOVES:
def player_not_on_turn_pieces_exception(request: ValidMovesPlayerRequest) -> None:

    if request.is_white_on_turn != request.is_from_white_player:
        raise PlayerNotOnTurnException()

def player_wants_moves_for_enemy_pieces_exception(request: ValidMovesPlayerRequest) -> None:

    if request.is_from_white_player != request.is_white:
        raise NotAllowedActionForEnemyPiecesException()


# PIECE_VALID_MOVES:
def player_not_on_turn_piece_exception(request: ValidMovesPieceRequest) -> None:

    if request.is_white_on_turn != request.is_from_white_player:
        raise PlayerNotOnTurnException()

def player_wants_moves_for_enemy_piece_exception(request: ValidMovesPieceRequest) -> None:

    if get_piece_by_uid(request.pieces, request.uid).is_white != request.is_from_white_player:
        raise NotAllowedActionForEnemyPiecesException()