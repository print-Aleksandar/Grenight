from domain.exceptions import (NonExistentValidMoveException,
                               PiecePinnedException,
                               TryingToTakeEnemyKingException)
from domain.pieces import Piece
from domain.requests import MoveRequest, ValidMovesPlayerRequest, ValidMovesPieceRequest
from domain.responses import MoveResponse, ValidMovesPlayerResponse, ValidMovesPieceResponse
from application.board_getter import all_per_move_getter, BoardGetter, get_piece_by_uid
from application.filters import filter_valid_attacks, filter_initial_moves
from application.moves import MoveRegistry


def filter_valid_player_moves(pieces: list[Piece],
                              board_getter: BoardGetter,
                              uids_with_initial_moves: dict[str, list[tuple[int, int]]],
                              is_for_white: bool) -> dict[str, list[tuple[int, int]]] | None:

    valid_moves = dict()

    for piece in pieces:
        if piece.is_white == is_for_white:
            initial_valid_moves_for_piece = [MoveRegistry(pieces,
                                                          piece.uid,
                                                          move,
                                                          board_getter.free_positions,
                                                          False,
                                                          promote_to=None)
                                   for move in uids_with_initial_moves[piece.uid]]

            valid_moves[piece.uid] = [move for move in initial_valid_moves_for_piece
                                      if move.move.is_move_valid]

    return valid_moves if any(len(moves) > 0 for moves in valid_moves.values()) else None


"""
def is_current_state_safe_for_player(pieces: list[Piece],
                                     uids_with_valid_attacks: dict[str, list[tuple[int, int]]],
                                     is_for_white: bool) -> bool:

    enemy_attacks = []
    [enemy_attacks.extend(valid_attacks) for uid, valid_attacks in uids_with_valid_attacks.items()
     if get_piece_by_uid(pieces, uid).is_white != is_for_white]
    enemy_attacks = list(set(enemy_attacks))

    return not any(piece for piece in pieces if piece.is_white == is_for_white
                    and not piece.can_be_on_attacked_position
                    and piece.position in enemy_attacks)
"""


def reject_move(request: MoveRequest,
                exception: str) -> MoveResponse:

    return MoveResponse(request.pieces, False, False, False,
                        request.is_white_on_turn, request.is_current_move_promotion,
                        None, exception)


def return_ongoing(request: MoveRequest,
                   move_registry: MoveRegistry) -> MoveResponse:

    return MoveResponse(move_registry.move.new_pieces, False, False, False,
                        not request.is_white_on_turn, move_registry.is_next_move_promotion,
                        move_registry.move.is_enemy_in_check, None)


def return_draw(request: MoveRequest, new_pieces: list[Piece]) -> MoveResponse:

    return MoveResponse(new_pieces, True, True, False,
                        not request.is_white_on_turn, False,
                        None, None)


def return_winner(request: MoveRequest, new_pieces: list[Piece]) -> MoveResponse:

    return MoveResponse(new_pieces, True, False, request.is_white_on_turn,
                        not request.is_white_on_turn, False)


def make_move(request: MoveRequest) -> MoveResponse:

    dummy = all_per_move_getter(request.pieces)
    current_board_getter = BoardGetter(dummy)
    current_uids_with_valid_attacks = filter_valid_attacks(current_board_getter)
    current_valid_moves = filter_initial_moves(request.pieces,
                                               current_uids_with_valid_attacks, current_board_getter)

    if not request.is_current_move_promotion and request.position not in current_valid_moves[request.uid]:
        return reject_move(request, NonExistentValidMoveException.__name__)


    try:
        move_registry = MoveRegistry(request.pieces, request.uid, request.position,
                            current_board_getter.free_positions, request.is_current_move_promotion,
                            request.promote_to)

    except TryingToTakeEnemyKingException:
        return reject_move(request, TryingToTakeEnemyKingException.__name__)

    if not move_registry.move.is_move_valid:
        return reject_move(request, PiecePinnedException.__name__)

    new_pieces = move_registry.move.new_pieces

    new_board_getter = BoardGetter(all_per_move_getter(new_pieces))
    new_uids_with_valid_attacks = filter_valid_attacks(new_board_getter)
    new_initial_valid_moves = filter_initial_moves(new_pieces,
                                                   new_uids_with_valid_attacks, new_board_getter)

    enemy_color = not request.is_white_on_turn

    enemy_next_valid_moves = filter_valid_player_moves(new_pieces, new_board_getter,
                                                       new_initial_valid_moves, enemy_color)

    # is_enemy_now_safe = is_current_state_safe_for_player(new_pieces, new_uids_with_valid_attacks, enemy_color)

    if enemy_next_valid_moves is None:
        if move_registry.move.is_enemy_in_check:
            return return_winner(request, new_pieces)
        return return_draw(request, new_pieces)
    return return_ongoing(request, move_registry)


def gather_valid_moves_color(request: ValidMovesPlayerRequest) -> ValidMovesPlayerResponse:

    dummy = all_per_move_getter(request.pieces)
    board_getter = BoardGetter(dummy)
    uids_with_valid_attacks = filter_valid_attacks(board_getter)
    valid_moves = filter_initial_moves(request.pieces,
                                               uids_with_valid_attacks, board_getter)
    color_valid_moves = dict()
    for uid, valid_moves in valid_moves.items():
        if get_piece_by_uid(request.pieces, uid).is_white == request.is_white:
            color_valid_moves[uid] = valid_moves

    return ValidMovesPlayerResponse(color_valid_moves)


def gather_valid_moves_piece(request: ValidMovesPieceRequest) -> ValidMovesPieceResponse:

    dummy = all_per_move_getter(request.pieces)
    board_getter = BoardGetter(dummy)
    uids_with_valid_attacks = filter_valid_attacks(board_getter)
    valid_moves = filter_initial_moves(request.pieces,
                                       uids_with_valid_attacks, board_getter)
    color_valid_moves = dict()
    for uid, moves in valid_moves.items():
        color_valid_moves[uid] = moves

    return ValidMovesPieceResponse(request.uid, valid_moves[request.uid])
