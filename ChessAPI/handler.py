from static import *
from services.game_service import *
from services.validator_service import *


#def is_threefold_repetition_draw(moves: list[tuple[str, tuple[int, int]]])


#def is_50_moves_draw(moves: list[tuple[str, tuple[int, int]]])


#def is_insufficient_material_draw(board: list[Piece])


#def get_game_status(board: list[Piece])


def rebuild_board(moves: list[tuple[str, tuple[int, int]]]) \
        -> list[Piece]:
    pieces = get_init_board()

    prev_piece_uid = None
    for uid, position in moves:
        if prev_piece_uid is not None:
            prev_piece = get_piece_by_uid(pieces, prev_piece_uid)
            if prev_piece is not None:
                if isinstance(prev_piece, Pawn):
                    prev_piece.is_el_passant_vulnerable = False

        piece = get_piece_by_uid(pieces, uid)
        pawn_result = is_pawn_en_passant_vulnerable(piece, position)
        pieces, new_piece, is_board_changed = apply_move(pieces, piece, position)
        if pawn_result and isinstance(new_piece, Pawn):
            new_piece.is_el_passant_vulnerable = True

        if not is_board_changed:
            raise InternalLoadingException()

        piece = get_piece_by_uid(pieces, uid)
        if not piece.had_first_move:
            piece.had_first_move = True

        prev_piece_uid = new_piece.uid

    return pieces


def play_move(pieces: list[Piece],
              requested_move: tuple[str, tuple[int, int]],
              is_from_white_player) \
        -> tuple[list[Piece], tuple[str, tuple[int, int]]]:


    uid = requested_move[0]
    piece = get_piece_by_uid(pieces, uid)
    if piece is None:
        raise InvalidRequestedMoveException()

    if piece.is_white != is_from_white_player:
        raise InvalidRequestedMoveException()

    position = tuple((int(requested_move[1][0]), int(requested_move[1][1])))
    if not is_position_within_board(position) or position not in piece.get_moving_positions(pieces):
        raise InvalidRequestedMoveException()

    new_pieces, _, is_board_changed = apply_move(pieces.copy(), piece, position)
    if not is_board_changed:
        raise InvalidRequestedMoveException()

    return new_pieces, tuple((uid, position))


def handler(moves: list[tuple[str, tuple[int, int]]],
            requested_move: tuple[str, tuple[int, int]],
            is_from_white_player: bool) -> dict:
    pieces = rebuild_board(moves)
    try:
        new_pieces, new_move = play_move(pieces, requested_move, is_from_white_player)
        moves.append(new_move)
        return get_game_status(pieces, moves, is_from_white_player)
    except InvalidRequestedMoveException:
        return {'pieces': pieces, 'moves': moves, 'status': 'ongoing', 'next_player':
                'white_player' if is_from_white_player else 'black_player'}