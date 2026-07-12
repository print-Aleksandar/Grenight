from abc import ABC, abstractmethod
from domain.pieces import Piece, Rook, Bishop, Knight, Queen
from domain.exceptions import NonExistentPiecePromotionException, NonExistentValidMoveException
from runtime_getters import get_piece_by_uid, get_piece_by_position


class Move(ABC):
    def __init__(self, pieces_with_valid_attacks: list[tuple[Piece, list[tuple[int, int]]]],
                 uid: str,
                 position: tuple[int, int],
                 valid_moves: list[tuple[int, int]],
                 free_positions: list[tuple[int, int]],
                 promote_to=None) -> None:

        self.pieces_with_valid_attacks = pieces_with_valid_attacks
        self.uid = uid
        self.position = position
        self.valid_moves = valid_moves
        self.free_positions = free_positions
        self.promote_to = promote_to
        self.white_attacks = self.gather_white_attacks()
        self.black_attacks = self.gather_black_attacks()
        self.white_pieces = self.gather_white_pieces()
        self.black_pieces = self.gather_black_pieces()
        self.is_white = self.is_piece_white()
        self.new_pieces = self.new_state()

    def gather_white_attacks(self) -> list[tuple[int, int]]:
        dummy = []
        [dummy.extend(piece_with_valid_attacks[1]) for piece_with_valid_attacks
         in self.pieces_with_valid_attacks if piece_with_valid_attacks[0].is_white]
        list(set(dummy))
        return dummy

    def gather_black_attacks(self) -> list[tuple[int, int]]:
        dummy = []
        [dummy.extend(piece_with_valid_attacks[1]) for piece_with_valid_attacks
         in self.pieces_with_valid_attacks if not piece_with_valid_attacks[0].is_white]
        list(set(dummy))
        return dummy

    def return_pieces(self) -> list[Piece]:
        return [piece_with_valid_attacks[0] for piece_with_valid_attacks in
                self.pieces_with_valid_attacks]

    def gather_white_pieces(self) -> list[Piece]:
        return [piece for piece in self.return_pieces() if piece.is_white]

    def gather_black_pieces(self) -> list[Piece]:
        return [piece for piece in self.return_pieces() if not piece.is_white]

    def call_piece_by_uid(self) -> Piece:
        return get_piece_by_uid(self.return_pieces(), self.uid)

    def gather_valid_attacks(self) -> list[tuple[int, int]]:
        return next((piece_with_valid_attacks[1] for piece_with_valid_attacks in
                     self.pieces_with_valid_attacks
                     if piece_with_valid_attacks[0].uid == self.uid))

    def is_piece_white(self) -> bool:
        return self.call_piece_by_uid().is_white

    @abstractmethod
    def supports(self) -> bool:
        pass

    @abstractmethod
    def new_state(self) -> list[Piece] | None:
        pass


class MoveRegistry:
    def __init__(self, pieces_with_valid_attacks: list[tuple[Piece, list[tuple[int, int]]]],
                 uid: str,
                 position: tuple[int, int],
                 valid_moves: list[tuple[int, int]],
                 free_positions: list[tuple[int, int]],
                 promote_to=None) -> None:

        self.pieces_with_valid_attacks = pieces_with_valid_attacks
        self.uid = uid
        self.position = position
        self.valid_moves = valid_moves
        self.free_positions = free_positions
        self.promote_to = promote_to

    def registry(self) -> list[Piece] | None:

        free_position_move = FreePositionMove(self.pieces_with_valid_attacks, self.uid,
                                              self.position, self.valid_moves,
                                              self.free_positions, self.promote_to)

        enemy_position_move = EnemyPositionMove(self.pieces_with_valid_attacks, self.uid,
                                                self.position, self.valid_moves,
                                                self.free_positions, self.promote_to)

        promotion_addition = PromotionAddition(self.pieces_with_valid_attacks, self.uid,
                                               self.position, self.valid_moves,
                                               self.free_positions, self.promote_to)

        pieces = None

        if free_position_move.supports():
            pieces = free_position_move.new_state()

        elif enemy_position_move.supports():
            pieces = enemy_position_move.new_state()

        if pieces is not None and promotion_addition.supports():
            pieces = promotion_addition.new_state()

        if pieces is None:
            raise NonExistentValidMoveException()
        return pieces


class FreePositionMove(Move):
    def __init__(self, pieces_with_valid_attacks: list[tuple[Piece, list[tuple[int, int]]]],
                 uid: str,
                 position: tuple[int, int],
                 free_positions: list[tuple[int, int]],
                 promote_to=None) -> None:

        super().__init__(pieces_with_valid_attacks, uid, position, free_positions, promote_to)

    def supports(self) -> bool:
        piece = self.call_piece_by_uid()
        return piece.position in self.free_positions

    def new_state(self) -> list[Piece]:
        self.call_piece_by_uid().position = self.position
        return self.return_pieces()


class EnemyPositionMove(Move):
    def __init__(self, pieces_with_valid_attacks: list[tuple[Piece, list[tuple[int, int]]]],
                 uid: str,
                 position: tuple[int, int],
                 free_positions: list[tuple[int, int]],
                 promote_to=None) -> None:

        super().__init__(pieces_with_valid_attacks, uid, position, free_positions, promote_to)

    def supports(self) -> bool:
        piece = self.call_piece_by_uid()
        attacked_piece = get_piece_by_position(self.return_pieces(), self.position)
        return piece.is_white != attacked_piece.is_white and attacked_piece.can_be_captured \
               if attacked_piece is not None else False

    def new_state(self) -> list[Piece]:
        piece = self.call_piece_by_uid()
        attacked_piece = get_piece_by_position(self.return_pieces(), self.position)
        piece.position = self.position
        return [piece for piece in self.return_pieces() if piece.uid != attacked_piece.uid]


class PromotionAddition(Move):
    def __init__(self, pieces_with_valid_attacks: list[tuple[Piece, list[tuple[int, int]]]],
                 uid: str,
                 position: tuple[int, int],
                 free_positions: list[tuple[int, int]],
                 promote_to=None) -> None:

        super().__init__(pieces_with_valid_attacks, uid, position, free_positions, promote_to)

    def supports(self) -> bool:
        piece = self.call_piece_by_uid()
        return piece.can_implement_pawn_moves and piece.is_next_move_pawn_promotion \
               and self.promote_to is not None

    def new_state(self) -> list[Piece]:
        piece = self.call_piece_by_uid()
        pieces = [piece for piece in self.return_pieces() if piece.uid != piece.uid]

        if self.promote_to == 'r':
            promoted_piece = Rook(uid=piece.uid, position=self.position)
        elif self.promote_to == 'b':
            promoted_piece = Bishop(uid=piece.uid, position=self.position)
        elif self.promote_to == 'k':
            promoted_piece = Knight(uid=piece.uid, position=self.position)
        elif self.promote_to == 'q':
            promoted_piece = Queen(uid=piece.uid, position=self.position)
        else:
            raise NonExistentPiecePromotionException()

        return pieces + [promoted_piece]
