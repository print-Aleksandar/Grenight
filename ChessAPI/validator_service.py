from pieces import Piece, Pawn
from game_service import *
from board_service import *


def is_position_within_board(position: tuple[int, int]) \
        -> bool:
    y, x = position
    return 0 <= y < 8 and 0 <= x < 8


def is_position_attacked_by_enemy(pieces: list[Piece],
                                  piece: Piece,
                                  position: tuple[int, int]) \
        -> bool:
    return position in get_enemy_attacking_positions(pieces, piece)


def is_position_attacked_by_ally(pieces: list[Piece],
                                  piece: Piece,
                                  position: tuple[int, int]) \
        -> bool:
    return position in get_ally_attacking_positions(pieces, piece)


def is_ally_king_safe(pieces: list[Piece],
                       piece: Piece) \
        -> bool:
    ally_king = get_ally_king(pieces, piece)
    return not is_position_attacked_by_enemy(pieces, piece, ally_king.position)


def is_enemy_king_safe(pieces: list[Piece],
                       piece: Piece) \
        -> bool:
    enemy_king = get_enemy_king(pieces, piece)
    return not is_position_attacked_by_ally(pieces, piece, enemy_king.position)


def is_move_valid(pieces: list[Piece],
                  piece: Piece,
                  position: tuple[int, int],
                  shall_be_in_positions: list[tuple[int, int]] | None) \
        -> bool:
    if shall_be_in_positions is not None:
        if position not in shall_be_in_positions:
            return False

    if get_enemy_king(pieces, piece).position == position:
        return False

    pieces_applied_move, piece_applied_move, is_board_changed = apply_move(pieces, piece, position)
    return is_ally_king_safe(pieces_applied_move, piece_applied_move) and is_board_changed


def is_enemy_king_stalemate(pieces: list[Piece],
                            piece: Piece) \
        -> bool:
    if not is_enemy_king_safe(pieces, piece):
        return False

    return len(get_enemy_valid_moving_positions(pieces, piece)) == 0


def is_enemy_king_checkmate(pieces: list[Piece],
                            piece: Piece) \
        -> bool:
    if is_enemy_king_safe(pieces, piece):
        return False

    return len(get_enemy_valid_moving_positions(pieces, piece)) == 0


def is_requested_move_castling(piece: Piece,
                               position: tuple[int, int]) \
        -> bool:
    if piece.uid != 'wk' or piece.uid != 'bk':
        return False

    return abs(piece.position[1] - position[1]) == 2 \
            and position[0] == piece.position[0]


def is_castling_possible(pieces: list[Piece],
                         piece: Piece,
                         is_left_castling: bool) \
        -> bool:
    king = get_ally_king(pieces, piece)
    if is_left_castling:
        rook = get_piece_by_uid(pieces, 'wlr') if piece.is_white \
            else get_piece_by_uid(pieces, 'blr')
    else:
        rook = get_piece_by_uid(pieces, 'wrr') if not piece.is_white \
            else get_piece_by_uid(pieces, 'brr')

    if king.had_first_move or rook.had_first_move:
        return False

    dx = [0, -1, -2] if is_left_castling else [0, 1, 2]
    for cx in dx:
        mp = (king.position[0], king.position[1] + cx)
        if mp not in get_free_positions(pieces, piece) or \
                is_position_attacked_by_enemy(pieces, piece, mp):
            return False

    return True


def is_pawn_en_passant_vulnerable(piece: Piece,
                                  position: tuple[int, int]) \
        -> bool:
    if not isinstance(piece, Pawn) or piece.had_first_move:
        return False
    else:
        return abs(piece.position[0] - position[0]) == 2 \
            and piece.position[1] == position[1]


def is_requested_move_en_passant(pieces: list[Piece],
                                 piece: Piece,
                                 position: tuple[int, int]) \
        -> bool:
    if not isinstance(piece, Pawn):
        return False

    if position not in get_free_positions(pieces, piece):
        return False

    sign = 1 if piece.is_white else -1
    return position in [(piece.position[0] + sign, piece.position[1] + 1),
                        (piece.position[0] + sign, piece.position[1] - 1)]


def is_en_passant_possible(pieces: list[Piece],
                           piece: Piece,
                           is_left_side) \
        -> bool:
    for enemy_pawn in get_enemy_pawns(pieces, piece):
        if isinstance(enemy_pawn, Pawn):
            if enemy_pawn.is_el_passant_vulnerable:
                if enemy_pawn.position[0] == piece.position[0]:
                    if is_left_side:
                        if enemy_pawn.position[1] < piece.position[1]:
                            return True
                    else:
                        if enemy_pawn.position[1] > piece.position[1]:
                            return True
    return False


def is_next_move_pawn_promotion(piece: Piece) \
        -> bool:
    if not isinstance(piece, Pawn):
        return False

    return piece.position[0] == 6 if piece.is_white else \
        piece.position[0] == 1