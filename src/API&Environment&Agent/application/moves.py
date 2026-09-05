from abc import ABC, abstractmethod
from domain.pieces import Piece, Rook, Queen, PIECES_CLASSES, Bishop, Knight, King, PIECES_VALUES
from domain.configs import ROWS, COLUMNS
from domain.exceptions import (NonExistentPiecePromotionException,
                               NonExistentValidMoveException,
                               TryingToTakeEnemyKingException)
from application.board_getter import BoardGetter, all_per_move_getter, get_piece_by_position


class Move(ABC):
    def __init__(self, current_pieces: list[Piece],
                 uid: str,
                 position: tuple[int, int],
                 current_free_positions: list[tuple[int, int]],
                 is_current_move_promotion: bool,
                 promote_to: int | None,
                 current_uid_index: dict[str, Piece] | None = None) -> None:

        self.current_pieces = current_pieces
        self.uid = uid
        self.position = position
        self.current_free_positions = current_free_positions
        self.is_current_move_promotion = is_current_move_promotion
        self.promote_to = promote_to
        self.current_uid_index = current_uid_index if current_uid_index is not None \
            else {p.uid: p for p in current_pieces}
        self.is_white = self.is_piece_white()

        # MAKING AND AFTER MOVE ATTRIBUTES:
        self.supports = self.func_supports()
        self.new_pieces = self.new_state()

        if self.supports:
            dummy = all_per_move_getter(self.new_pieces)
            self.new_board_getter = BoardGetter(dummy)

            self.new_white_pieces = self.gather_new_white_pieces()
            self.new_black_pieces = self.gather_new_black_pieces()
            self.new_ally_pieces = self.gather_new_ally_pieces()
            self.new_enemy_pieces = self.gather_new_enemy_pieces()

        self.is_move_valid = self.is_ally_king_safe() if self.supports else False
        self.is_enemy_in_check = self.is_enemy_king_in_check() if self.supports and self.is_move_valid else None


    def gather_new_white_pieces(self) -> list[Piece]:
        return [piece for piece in self.new_pieces if piece.is_white]

    def gather_new_black_pieces(self) -> list[Piece]:
        return [piece for piece in self.new_pieces if not piece.is_white]

    def gather_new_ally_pieces(self) -> list[Piece]:
        return self.gather_new_white_pieces() if self.is_white else self.gather_new_black_pieces()

    def gather_new_enemy_pieces(self) -> list[Piece]:
        return self.gather_new_black_pieces() if self.is_white else self.gather_new_white_pieces()

    def is_piece_white(self) -> bool:
        return self.current_uid_index[self.uid].is_white

    def is_ally_king_safe(self) -> bool:
        return not self.is_king_in_check(self.is_white)

    def is_enemy_king_in_check(self) -> bool:
        return self.is_king_in_check(not self.is_white)

    def is_king_in_check(self, is_for_white: bool) -> bool:
        king = self.new_board_getter.get_king(is_for_white)
        y, x = king.position

        # CHECK FOR DIRECTIONAL ATTACKS:
        for i in range(y + 1, ROWS):
            pos = (i, x)
            piece = self.new_board_getter.get_piece_by_position(pos)

            if piece is None:
                continue

            if piece.is_white == is_for_white:
                break

            if piece.is_white != is_for_white:
                if pos in piece.attacking_positions:
                    continue

                if type(piece) in [Rook, Queen]:
                    return True
                else:
                    break

            return True

        for i in range(y - 1, -1, -1):
            pos = (i, x)
            piece = self.new_board_getter.get_piece_by_position(pos)

            if piece is None:
                continue

            if piece.is_white == is_for_white:
                break

            if piece.is_white != is_for_white:
                if pos in piece.attacking_positions:
                    continue

                if type(piece) in [Rook, Queen]:
                    return True
                else:
                    break

            return True

        for i in range(x + 1, COLUMNS):
            pos = (y, i)
            piece = self.new_board_getter.get_piece_by_position(pos)

            if piece is None:
                continue

            if piece.is_white == is_for_white:
                break

            if piece.is_white != is_for_white:
                if pos in piece.attacking_positions:
                    continue

                if type(piece) in [Rook, Queen]:
                    return True
                else:
                    break

            return True

        for i in range(x - 1, -1, - 1):
            pos = (y, i)
            piece = self.new_board_getter.get_piece_by_position(pos)

            if piece is None:
                continue

            if piece.is_white == is_for_white:
                break

            if piece.is_white != is_for_white:
                if pos in piece.attacking_positions:
                    continue

                if type(piece) in [Rook, Queen]:
                    return True
                else:
                    break

            return True

        # CHECK FOR PAWN ATTACKS:
        if self.new_board_getter.is_pos_attacked_by_color_pawn(not king.is_white, king.position):
            return True

        # CHECK FOR DIAGONAL ATTACKS:
        for i in range(1, min(ROWS - y, COLUMNS - x)):
            pos = (y + i, x + i)
            piece = self.new_board_getter.get_piece_by_position(pos)

            if piece is None:
                continue

            if piece.is_white == is_for_white:
                break

            if piece.is_white != is_for_white:
                if pos in piece.attacking_positions:
                    continue

                if type(piece) in [Bishop, Queen]:
                    return True
                else:
                    break

            return True

        for i in range(1, min(ROWS - y, x + 1)):
            pos = (y + i, x - i)
            piece = self.new_board_getter.get_piece_by_position(pos)

            if piece is None:
                continue

            if piece.is_white == is_for_white:
                break

            if piece.is_white != is_for_white:
                if pos in piece.attacking_positions:
                    continue

                if type(piece) in [Bishop, Queen]:
                    return True
                else:
                    break

            return True

        for i in range(1, min(y + 1, COLUMNS - x)):
            pos = (y - i, x + i)
            piece = self.new_board_getter.get_piece_by_position(pos)

            if piece is None:
                continue

            if piece.is_white == is_for_white:
                break

            if piece.is_white != is_for_white:
                if pos in piece.attacking_positions:
                    continue

                if type(piece) in [Bishop, Queen]:
                    return True
                else:
                    break

            return True

        for i in range(1, min(y + 1, x + 1)):
            pos = (y - i, x - i)
            piece = self.new_board_getter.get_piece_by_position(pos)

            if piece is None:
                continue

            if piece.is_white == is_for_white:
                break

            if piece.is_white != is_for_white:
                if pos in piece.attacking_positions:
                    continue

                if type(piece) in [Bishop, Queen]:
                    return True
                else:
                    break

            return True

        # CHECK FOR KING ATTACKS
        dyx = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, 1), (1, -1), (-1, -1)]
        positions = [(y + dy, x + dx) for (dy, dx) in dyx]
        for pos in positions:
            piece = self.new_board_getter.get_piece_by_position(pos)

            if piece is None:
                continue

            if piece.is_white != is_for_white and type(piece) in [King]:
                return True


        # CHECK FOR KNIGHT ATTACKS
        dyx = [(2, 1), (1, 2), (2, -1), (1, -2), (-2, 1), (-1, 2), (-2, -1), (-1, -2)]
        positions = [(y + dy, x + dx) for (dy, dx) in dyx]
        for pos in positions:
            piece = self.new_board_getter.get_piece_by_position(pos)

            if piece is None:
                continue

            if piece.is_white != is_for_white and type(piece) in [Knight]:
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
                 promote_to: int | None,
                 current_uid_index: dict[str, Piece] | None = None) -> None:

        self.current_pieces = current_pieces
        self.uid = uid
        self.position = position
        self.current_free_positions = current_free_positions
        self.is_current_move_promotion = is_current_move_promotion
        self.promote_to = promote_to
        self.current_uid_index = current_uid_index if current_uid_index is not None \
            else {p.uid: p for p in current_pieces}

        self.attacked_piece_value = None

        self.move = self.registry()
        self.is_next_move_promotion = self.is_next_promotable()


    def registry(self) -> Move:

        free_position_move = FreePositionMove(self.current_pieces, self.uid,
                                              self.position, self.current_free_positions,
                                              self.is_current_move_promotion, self.promote_to,
                                              self.current_uid_index)

        enemy_position_move = EnemyPositionMove(self.current_pieces, self.uid,
                                                self.position, self.current_free_positions,
                                                self.is_current_move_promotion, self.promote_to,
                                                self.current_uid_index)

        finishing_promotion_move = FinishingPromotionMove(self.current_pieces, self.uid,
                                                          self.position, self.current_free_positions,
                                                          self.is_current_move_promotion, self.promote_to,
                                                          self.current_uid_index)

        if self.is_current_move_promotion:
            if finishing_promotion_move.supports:
                return finishing_promotion_move
            else:
                raise NonExistentValidMoveException()

        else:
            if free_position_move.supports:
                return free_position_move

            if enemy_position_move.supports:
                self.attacked_piece_value = enemy_position_move.attacked_piece_value
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
                 promote_to: int | None,
                 current_uid_index: dict[str, Piece] | None = None) -> None:

        super().__init__(current_pieces, uid, position,
                         current_free_positions, is_current_move_promotion, promote_to,
                         current_uid_index)

    def new_state(self) -> list[Piece] | None:
        if not self.supports:
            return None

        moved_piece = self.current_uid_index[self.uid]
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
                 promote_to: int | None,
                 current_uid_index: dict[str, Piece] | None = None) -> None:

        self.attacked_piece_value = None

        super().__init__(current_pieces, uid, position,
                         current_free_positions, is_current_move_promotion, promote_to,
                         current_uid_index)

    def new_state(self) -> list[Piece] | None:
        if not self.supports:
            return None

        moved_piece = self.current_uid_index[self.uid]
        attacked_piece = get_piece_by_position(self.current_pieces, self.position)
        if not attacked_piece.can_be_captured:
            raise TryingToTakeEnemyKingException()
        new_piece = moved_piece.moved_to(self.position)
        return [new_piece if p.uid == self.uid else p
                for p in self.current_pieces if p.uid != attacked_piece.uid]

    def func_supports(self) -> bool:
        piece = self.current_uid_index[self.uid]
        attacked_piece = get_piece_by_position(self.current_pieces, self.position)
        if attacked_piece is None:
            return False
        self.attacked_piece_value = PIECES_VALUES[type(attacked_piece)]
        return piece.is_white != attacked_piece.is_white \
               if attacked_piece is not None else False


class PromotionAddition(Move):
    def __init__(self, current_pieces: list[Piece],
                 uid: str,
                 position: tuple[int, int],
                 current_free_positions: list[tuple[int, int]],
                 is_current_move_promotion: bool,
                 promote_to: int | None,
                 current_uid_index: dict[str, Piece] | None = None) -> None:

        super().__init__(current_pieces, uid, position,
                         current_free_positions, is_current_move_promotion, promote_to,
                         current_uid_index)

    def new_state(self) -> list[Piece] | None:
        return self.current_pieces

    def func_supports(self) -> bool:
        piece = self.current_uid_index[self.uid]
        return piece.can_implement_pawn_moves and piece.is_current_move_pawn_promotion \
                    and self.promote_to is None


class FinishingPromotionMove(Move):
    def __init__(self, current_pieces: list[Piece],
                 uid: str,
                 position: tuple[int, int],
                 current_free_positions: list[tuple[int, int]],
                 is_current_move_promotion: bool,
                 promote_to: int | None,
                 current_uid_index: dict[str, Piece] | None = None) -> None:

        super().__init__(current_pieces, uid, position,
                         current_free_positions, is_current_move_promotion, promote_to,
                         current_uid_index)

    def func_supports(self) -> bool:
        piece = self.current_uid_index[self.uid]
        return self.is_current_move_promotion \
            and piece.can_implement_pawn_moves and piece.is_current_move_pawn_promotion \
            and piece.position == self.position

    def new_state(self) -> list[Piece] | None:
        if not self.supports:
            return None

        piece = self.current_uid_index[self.uid]
        pieces = [piece for piece in self.current_pieces if self.uid != piece.uid]
        cls = PIECES_CLASSES.get(self.promote_to, None)

        if cls is not None and cls in [Rook, Queen]:
            promoted_piece = cls(uid=self.uid, is_white=self.is_white,
                                 position=self.position, had_first_move=piece.had_first_move)
        else:
            raise NonExistentPiecePromotionException()

        return pieces + [promoted_piece]
