from pieces import Piece, Pawn, Knight, Bishop, Rook, Queen, King


POSITIONS = [(num // 8, num % 8) for num in range(64)]


class InvalidRequestedMoveException(Exception):
    pass


class InternalLoadingException(Exception):
    pass


def get_init_board() -> list[Piece]:
    return [Rook('wlr', True, (0, 0)),
            Knight('wlk', True, (0, 1)),
            Bishop('wlb', True, (0, 2)),
            Queen('wq', True, (0, 3)),
            King('wk', True, (0, 4)),
            Bishop('wrb', True, (0, 5)),
            Knight('wrk', True, (0, 6)),
            Rook('wrr', True, (0, 7)),

            Pawn('wp0', True, (1, 0)),
            Pawn('wp1', True, (1, 1)),
            Pawn('wp2', True, (1, 2)),
            Pawn('wp3', True, (1, 3)),
            Pawn('wp4', True, (1, 4)),
            Pawn('wp5', True, (1, 5)),
            Pawn('wp6', True, (1, 6)),
            Pawn('wp7', True, (1, 7)),

            Rook('blr', False, (7, 0)),
            Knight('blk', False, (7, 1)),
            Bishop('blb', False, (7, 2)),
            Queen('bq', False, (7, 3)),
            King('bk', False, (7, 4)),
            Bishop('brb', False, (7, 5)),
            Knight('brk', False, (7, 6)),
            Rook('brr', False, (7, 7)),

            Pawn('bp0', False, (6, 0)),
            Pawn('bp1', False, (6, 1)),
            Pawn('bp2', False, (6, 2)),
            Pawn('bp3', False, (6, 3)),
            Pawn('bp4', False, (6, 4)),
            Pawn('bp5', False, (6, 5)),
            Pawn('bp6', False, (6, 6)),
            Pawn('bp7', False, (6, 7))]