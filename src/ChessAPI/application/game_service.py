from copy import deepcopy

from domain.exceptions import NonExistentValidMoveException, NonExistentPiecePromotionException
from domain.pieces import Piece
from application.runtime_getters import all_per_move_getter
from application.runtime_filter import filter_valid_moves
from application.moves import MoveRegistry


def pseudo_move(pieces: list[Piece],
                uid: str,
                position: tuple[int, int],
                promote_to=None) -> list[Piece]:

    pieces_cpy = deepcopy(pieces)

    white_pieces, black_pieces, white_positions, \
    black_positions, free_positions = all_per_move_getter(pieces_cpy)

    pieces_attack_moves = filter_valid_moves(white_pieces, black_pieces, white_positions,
                                            black_positions, free_positions)

    pieces_with_valid_attacks = [(piece_attack_moves[0], pieces_attack_moves[1])
                                 for piece_attack_moves in pieces_attack_moves]

    pieces_with_valid_moves = [(piece_attack_moves[0], pieces_attack_moves[2])
                                 for piece_attack_moves in pieces_attack_moves]

    try:
        candidate_new_state = MoveRegistry(pieces_with_valid_attacks, uid, position, promote_to).registry()
    except NonExistentValidMoveException as e:
        # return move invalid
        pass