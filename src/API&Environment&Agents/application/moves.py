from abc import ABC, abstractmethod
from domain.pieces import Piece, Knight, Bishop, Rook, Queen
from domain.board_initialization import PIECES_CLASSES
from domain.exceptions import NonExistentPiecePromotionException, NonExistentValidMoveException
from application.board_getter import BoardGetter, all_per_move_getter, get_piece_by_uid, get_piece_by_position
from application.filters import filter_valid_attacks


class Move(ABC):
    def __init__(self, current_pieces: list[Piece],
                 uid: str,
                 position: tuple[int, int],
                 current_free_positions: list[tuple[int, int]],
                 is_current_move_promotion: bool,
                 promote_to: int | None) -> None:

        self.current_pieces = current_pieces
        self.uid = uid
        self.position = position
        self.current_free_positions = current_free_positions
        self.is_current_move_promotion = is_current_move_promotion
        self.promote_to = promote_to
        self.is_white = self.is_piece_white()

        # MAKING AND AFTER MOVE ATTRIBUTES:
        self.supports = self.func_supports()
        self.new_pieces = self.new_state()

        if self.supports:
            dummy = all_per_move_getter(self.new_pieces)
            self.new_board_getter = BoardGetter(dummy)
            self.new_uids_with_valid_attacks = filter_valid_attacks(self.new_board_getter)
            self.new_white_attacks = self.gather_new_white_attacks()
            self.new_black_attacks = self.gather_new_black_attacks()
            self.new_enemy_attacks = self.gather_new_enemy_attacks()
            self.new_ally_attacks = self.gather_new_ally_attacks()

            self.new_white_pieces = self.gather_new_white_pieces()
            self.new_black_pieces = self.gather_new_black_pieces()
            self.new_ally_pieces = self.gather_new_ally_pieces()
            self.new_enemy_pieces = self.gather_new_enemy_pieces()

        self.is_move_valid = self.is_ally_king_safe() if self.supports else False
        self.is_enemy_in_check = self.is_enemy_king_in_check() if self.supports and self.is_move_valid else None

    def gather_new_white_attacks(self) -> list[tuple[int, int]]:
        dummy = []
        [dummy.extend(valid_attacks) for uid, valid_attacks
         in self.new_uids_with_valid_attacks.items()
         if get_piece_by_uid(self.new_pieces, uid).is_white]
        dummy = list(set(dummy))
        return dummy

    def gather_new_black_attacks(self) -> list[tuple[int, int]]:
        dummy = []
        [dummy.extend(valid_attacks) for uid, valid_attacks
         in self.new_uids_with_valid_attacks.items()
         if not get_piece_by_uid(self.new_pieces, uid).is_white]
        dummy = list(set(dummy))
        return dummy

    def gather_new_enemy_attacks(self) -> list[tuple[int, int]]:
        return self.gather_new_black_attacks() if self.is_white else self.gather_new_white_attacks()

    def gather_new_ally_attacks(self) -> list[tuple[int, int]]:
        return self.gather_new_white_attacks() if self.is_white else self.gather_new_black_attacks()

    def gather_new_white_pieces(self) -> list[Piece]:
        return [piece for piece in self.new_pieces if piece.is_white]

    def gather_new_black_pieces(self) -> list[Piece]:
        return [piece for piece in self.new_pieces if not piece.is_white]

    def gather_new_ally_pieces(self) -> list[Piece]:
        return self.gather_new_white_pieces() if self.is_white else self.gather_new_black_pieces()

    def gather_new_enemy_pieces(self) -> list[Piece]:
        return self.gather_new_black_pieces() if self.is_white else self.gather_new_white_pieces()

    def is_piece_white(self) -> bool:
        return get_piece_by_uid(self.current_pieces, self.uid).is_white

    def is_ally_king_safe(self) -> bool:
        for ally_piece in self.new_ally_pieces:
            if not ally_piece.can_be_on_attacked_position and \
                ally_piece.position in self.new_enemy_attacks:
                    return False
        return True

    def is_enemy_king_in_check(self) -> bool:
        for enemy_piece in self.new_enemy_pieces:
            if not enemy_piece.can_be_on_attacked_position and \
                enemy_piece.position in self.new_ally_attacks:
                return True
        return False

    @abstractmethod
    def new_state(self) -> list[Piece] | None:
        pass

    @abstractmethod
    def func_supports(self) -> bool:
        pass


class MoveRegistry:
    def __init__(self, current_pieces: list[Piece],
                 uid: str,
                 position: tuple[int, int],
                 current_free_positions: list[tuple[int, int]],
                 is_current_move_promotion: bool,
                 promote_to: int | None) -> None:

        self.current_pieces = current_pieces
        self.uid = uid
        self.position = position
        self.current_free_positions = current_free_positions
        self.is_current_move_promotion = is_current_move_promotion
        self.promote_to = promote_to

        self.move = self.registry()
        self.is_next_move_promotion = self.is_next_promotable()

    def registry(self) -> Move:

        free_position_move = FreePositionMove(self.current_pieces, self.uid,
                                              self.position, self.current_free_positions,
                                              self.is_current_move_promotion, self.promote_to)

        enemy_position_move = EnemyPositionMove(self.current_pieces, self.uid,
                                                self.position, self.current_free_positions,
                                                self.is_current_move_promotion, self.promote_to)

        finishing_promotion_move = FinishingPromotionMove(self.current_pieces, self.uid,
                                                          self.position, self.current_free_positions,
                                                          self.is_current_move_promotion, self.promote_to)

        if self.is_current_move_promotion:
            if finishing_promotion_move.supports:
                return finishing_promotion_move
            else:
                raise NonExistentValidMoveException()

        else:
            if free_position_move.supports:
                return free_position_move

            if enemy_position_move.supports:
                return enemy_position_move

        raise NonExistentValidMoveException()

    def is_next_promotable(self) -> bool:
        return PromotionAddition(self.move.new_pieces, self.uid,
                                 self.position, self.current_free_positions,
                                 self.is_current_move_promotion, self.promote_to).func_supports()


class FreePositionMove(Move):
    def __init__(self, current_pieces: list[Piece],
                 uid: str,
                 position: tuple[int, int],
                 current_free_positions: list[tuple[int, int]],
                 is_current_move_promotion: bool,
                 promote_to: int | None) -> None:

        super().__init__(current_pieces, uid, position,
                         current_free_positions, is_current_move_promotion, promote_to)

    def new_state(self) -> list[Piece] | None:
        if not self.supports:
            return None

        moved_piece = get_piece_by_uid(self.current_pieces, self.uid)
        new_piece = moved_piece.moved_to(self.position)
        return [new_piece if p.uid == self.uid else p for p in self.current_pieces]

    def func_supports(self) -> bool:
        return self.position in self.current_free_positions

class EnemyPositionMove(Move):
    def __init__(self, current_pieces: list[Piece],
                 uid: str,
                 position: tuple[int, int],
                 current_free_positions: list[tuple[int, int]],
                 is_current_move_promotion: bool,
                 promote_to: int | None) -> None:

        super().__init__(current_pieces, uid, position,
                         current_free_positions, is_current_move_promotion, promote_to)

    def new_state(self) -> list[Piece] | None:
        if not self.supports:
            return None

        moved_piece = get_piece_by_uid(self.current_pieces, self.uid)
        attacked_piece = get_piece_by_position(self.current_pieces, self.position)
        new_piece = moved_piece.moved_to(self.position)
        return [new_piece if p.uid == self.uid else p
                for p in self.current_pieces if p.uid != attacked_piece.uid]

    def func_supports(self) -> bool:
        piece = get_piece_by_uid(self.current_pieces, self.uid)
        attacked_piece = get_piece_by_position(self.current_pieces, self.position)
        if attacked_piece is None:
            return False
        return piece.is_white != attacked_piece.is_white \
               if attacked_piece is not None else False


class PromotionAddition(Move):
    def __init__(self, current_pieces: list[Piece],
                 uid: str,
                 position: tuple[int, int],
                 current_free_positions: list[tuple[int, int]],
                 is_current_move_promotion: bool,
                 promote_to: int | None) -> None:

        super().__init__(current_pieces, uid, position,
                         current_free_positions, is_current_move_promotion, promote_to)

    def new_state(self) -> list[Piece] | None:
        return self.current_pieces

    def func_supports(self) -> bool:
        piece = get_piece_by_uid(self.current_pieces, self.uid)
        return piece.can_implement_pawn_moves and piece.is_current_move_pawn_promotion \
                    and self.promote_to is None


class FinishingPromotionMove(Move):
    def __init__(self, current_pieces: list[Piece],
                 uid: str,
                 position: tuple[int, int],
                 current_free_positions: list[tuple[int, int]],
                 is_current_move_promotion: bool,
                 promote_to: int | None) -> None:

        super().__init__(current_pieces, uid, position,
                         current_free_positions, is_current_move_promotion, promote_to)

    def func_supports(self) -> bool:
        piece = get_piece_by_uid(self.current_pieces, self.uid)
        return self.is_current_move_promotion \
            and piece.can_implement_pawn_moves and piece.is_current_move_pawn_promotion \
            and piece.position == self.position

    def new_state(self) -> list[Piece] | None:
        if not self.supports:
            return None

        piece = get_piece_by_uid(self.current_pieces, self.uid)
        pieces = [piece for piece in self.current_pieces if self.uid != piece.uid]
        cls = PIECES_CLASSES.get(self.promote_to, None)

        if cls is not None and cls in [Knight, Bishop, Rook, Queen]:
            promoted_piece = cls(uid=self.uid, is_white=self.is_white,
                                 position=self.position, had_first_move=piece.had_first_move)
        else:
            raise NonExistentPiecePromotionException()

        return pieces + [promoted_piece]
