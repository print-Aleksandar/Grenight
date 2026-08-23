from domain.pieces import Piece
from domain.exceptions import NonExistentValidMoveException, PiecePinnedException
from domain.requests import MoveRequest, ValidMovesPieceRequest, AgentMoveRequest
from domain.responses import MoveResponse, ValidMovesPieceResponse
from application.board_getter import all_per_move_getter, BoardGetter, get_piece_by_uid
from application.filters import filter_valid_attacks, filter_initial_moves
from application.moves import MoveRegistry


def is_player_having_any_valid_moves(pieces: list[Piece],
                                     board_getter: BoardGetter,
                                     uids_with_initial_moves: dict[str, list[tuple[int, int]]],
                                     is_for_white: bool) -> bool:

    for piece in pieces:
        if piece.is_white != is_for_white:
            continue

        for move in uids_with_initial_moves[piece.uid]:
            move_registry = MoveRegistry(pieces,
                                         piece.uid,
                                         move,
                                         board_getter.free_positions,
                                         is_current_move_promotion=False,
                                         promote_to=None)
            if move_registry.move.is_move_valid:
                return True

    return False


def return_ongoing(request: MoveRequest,
                   move_registry: MoveRegistry) -> MoveResponse:

    return MoveResponse(move_registry.move.new_pieces,
                        False,
                        False,
                        False,
                        request.is_white_on_turn if move_registry.is_next_move_promotion else not request.is_white_on_turn,
                        move_registry.is_next_move_promotion,
                        move_registry.move.is_enemy_in_check)


def return_draw(request: MoveRequest, new_pieces: list[Piece]) -> MoveResponse:

    return MoveResponse(new_pieces, True, True, False,
                        not request.is_white_on_turn, False)


def return_winner(request: MoveRequest, new_pieces: list[Piece]) -> MoveResponse:

    return MoveResponse(new_pieces, True, False, request.is_white_on_turn,
                        not request.is_white_on_turn, False)


def make_move(request: MoveRequest) -> MoveResponse:

    dummy = all_per_move_getter(request.pieces)
    current_board_getter = BoardGetter(dummy)

    current_uids_with_valid_attacks = filter_valid_attacks(current_board_getter)

    current_initial_valid_moves = filter_initial_moves(request.pieces,
                                                       current_uids_with_valid_attacks, current_board_getter)

    if not request.is_current_move_promotion and request.position not in current_initial_valid_moves[request.uid]:
        raise NonExistentValidMoveException()

    move_registry = MoveRegistry(request.pieces, request.uid, request.position,
                                 current_board_getter.free_positions, request.is_current_move_promotion,
                                 request.promote_to)

    if not move_registry.move.is_move_valid:
        raise PiecePinnedException()

    new_pieces = move_registry.move.new_pieces

    new_board_getter = BoardGetter(all_per_move_getter(new_pieces))

    new_uids_with_valid_attacks = filter_valid_attacks(new_board_getter)

    new_initial_valid_moves = filter_initial_moves(new_pieces,
                                                   new_uids_with_valid_attacks, new_board_getter)

    enemy_color = not request.is_white_on_turn

    enemy_next_valid_moves = is_player_having_any_valid_moves(new_pieces, new_board_getter,
                                                              new_initial_valid_moves, enemy_color)

    if not enemy_next_valid_moves:
        if move_registry.move.is_enemy_in_check:
            return return_winner(request, new_pieces)
        return return_draw(request, new_pieces)
    return return_ongoing(request, move_registry)


def gather_valid_moves_piece(request: ValidMovesPieceRequest) -> ValidMovesPieceResponse:

    dummy = all_per_move_getter(request.pieces)
    board_getter = BoardGetter(dummy)

    uids_with_valid_attacks = filter_valid_attacks(board_getter)

    uids_with_initial_moves = filter_initial_moves(request.pieces,
                                                   uids_with_valid_attacks, board_getter)

    initial_moves_registries_for_requested_piece = [MoveRegistry(request.pieces,
                                                               request.uid,
                                                               position,
                                                               board_getter.free_positions,
                                                               False,
                                                               promote_to=None)
                                                    for position in uids_with_initial_moves[request.uid]]

    valid_moves_registries_for_requested_piece = [move_registry for move_registry
                                                  in initial_moves_registries_for_requested_piece
                                                  if move_registry.move.is_move_valid]
    positions = []
    for move_registry in valid_moves_registries_for_requested_piece:
        positions.append(move_registry.move.position)

    return ValidMovesPieceResponse(request.uid, positions)


def gather_valid_moves_player(request: AgentMoveRequest) -> dict[str, list[tuple[int, int]]]:
    dummy = all_per_move_getter(request.pieces)
    board_getter = BoardGetter(dummy)

    uids_with_valid_attacks = filter_valid_attacks(board_getter)

    uids_with_initial_moves = filter_initial_moves(request.pieces,
                                                   uids_with_valid_attacks, board_getter)

    uids_with_valid_moves = dict()
    for uid, initial_moves in uids_with_initial_moves.items():
        if get_piece_by_uid(request.pieces, uid).is_white != request.is_for_white:
            continue

        pre_valid_moves = [MoveRegistry(request.pieces,
                                        uid,
                                        position,
                                        board_getter.free_positions,
                                        False,
                                        promote_to=None)
                       for position in initial_moves]

        valid_moves = [move_registry.move.position for move_registry in pre_valid_moves
                       if move_registry.move.is_move_valid]

        if len(valid_moves) > 0:
            uids_with_valid_moves[uid] = valid_moves

    return uids_with_valid_moves
