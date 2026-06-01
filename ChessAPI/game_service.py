from pieces import Piece, Pawn
from services.validator_service import *
from services.board_service import *


def apply_move(pieces: list[Piece],
               piece: Piece,
               position: tuple[int, int]) \
    -> tuple[list[Piece], Piece, bool]:
    is_board_changed = False

    if position in get_enemy_positions(pieces, piece):
        attacked_piece = get_piece_by_position(pieces, position)
        pieces.remove(attacked_piece)
        piece.position = position
        is_board_changed = True

    elif position in get_free_positions(pieces, piece):
        if is_requested_move_castling(piece, position):
            if is_castling_possible(pieces, piece, position[1] < piece.position[1]):
                pieces = apply_castling_move(pieces, piece, position[1] < piece.position[1])
                is_board_changed = True

        elif is_requested_move_en_passant(pieces, piece, position):
            if is_en_passant_possible(pieces, piece, position[1] < piece.position[1]):
                pieces, is_board_changed = apply_en_passant_move(pieces, piece, position[1] < piece.position[1])

        else:
            piece.position = position
            is_board_changed = True

    piece = get_piece_by_uid(pieces, piece.uid)
    return pieces, piece, is_board_changed


def apply_castling_move(pieces: list[Piece],
                        piece: Piece,
                        is_left_castling: bool) \
        -> list[Piece]:
    king = get_ally_king(pieces, piece)
    if is_left_castling:
        rook = get_piece_by_uid(pieces, 'wlr') if piece.is_white \
            else get_piece_by_uid(pieces, 'blr')
    else:
        rook = get_piece_by_uid(pieces, 'wrr') if piece.is_white \
            else get_piece_by_uid(pieces, 'brr')

    king_new_x = -2 if is_left_castling else 2
    rook_new_x = king.position[1] - 1 if is_left_castling else king.position[1] + 1

    king.position = (king.position[0], king_new_x)
    rook.position = (rook.position[0], rook_new_x)

    return pieces


def apply_en_passant_move(pieces: list[Piece],
                          piece: Piece,
                          is_en_passant_left: bool) \
        -> tuple[list[Piece], bool]:
    if not isinstance(piece, Pawn):
        return pieces, False

    dy = 1 if piece.is_white else -1
    dx = -1 if is_en_passant_left else 1

    position = (piece.position[0] + dy, piece.position[1] + dx)
    enemy_pawn = get_piece_by_position(pieces, position)
    if enemy_pawn is None:
        return pieces, False

    pieces.remove(enemy_pawn)
    pawn = get_piece_by_uid(pieces, piece.uid)
    pawn.position = position

    return pieces, True