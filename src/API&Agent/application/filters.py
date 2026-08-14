from application.board_getter import BoardGetter
from domain.pieces import Piece
from domain.board_configs import ROWS, COLUMNS


def filter_valid_attacks(board_getter: BoardGetter) \
        -> dict[str, list[tuple[int, int]]]:

     non_white_positions = board_getter.black_positions + board_getter.free_positions
     non_black_positions = board_getter.white_positions + board_getter.free_positions

     uids_with_valid_attacks = dict()

     for piece in board_getter.white_pieces:
          current_attacking_positions = piece.attacking_positions

          if not piece.is_moving_sequence:
               current_attacking_positions = [position for position in current_attacking_positions
                                              if position in non_white_positions]

          else:
               if piece.do_attacking_position_requires_enemy_on_it:
                    current_attacking_positions = [position for position in current_attacking_positions
                                                   if position in board_getter.black_positions]
               else:
                    current_filtered = []

                    y, x = piece.position
                    neighs = [neigh for neigh in current_attacking_positions
                              if (abs(neigh[0] - y) == 0 or abs(neigh[0] - y) == 1)
                              and (abs(neigh[1] - x) == 0 or abs(neigh[1] - x) == 1)]

                    dyx = [(neigh[0] - y, neigh[1] - x) for neigh in neighs]
                    for dy, dx in dyx:

                         if dy < 0:
                              y_range = range(0, -ROWS, -1)
                         elif dy > 0:
                              y_range = range(0, ROWS)
                         else:
                              y_range = [0 for _ in range(ROWS)]

                         if dx < 0:
                              x_range = range(0, -COLUMNS, -1)
                         elif dx > 0:
                              x_range = range(0, COLUMNS)
                         else:
                              x_range = [0 for _ in range(COLUMNS)]

                         for cy, cx in zip(y_range, x_range):
                              pos = (y + cy, x + cx)

                              if pos != piece.position:
                                   if pos not in piece.attacking_positions:
                                        break

                                   if pos in board_getter.white_positions:
                                        break

                                   if pos in board_getter.black_positions:
                                        current_filtered.append(pos)
                                        break

                                   if pos in board_getter.free_positions:
                                        current_filtered.append(pos)

                         current_attacking_positions = current_filtered

          uids_with_valid_attacks[piece.uid] = current_attacking_positions

     for piece in board_getter.black_pieces:
          current_attacking_positions = piece.attacking_positions
          if not piece.is_moving_sequence:
               current_attacking_positions = [position for position in current_attacking_positions
                                              if position in non_black_positions]

          else:
               if piece.do_attacking_position_requires_enemy_on_it:
                    current_attacking_positions = [position for position in current_attacking_positions
                                                   if position in board_getter.white_positions]
               else:
                    current_filtered = []

                    y, x = piece.position
                    neighs = [neigh for neigh in current_attacking_positions
                              if (abs(neigh[0] - y) == 0 or abs(neigh[0] - y) == 1)
                              and (abs(neigh[1] - x) == 0 or abs(neigh[1] - x) == 1)]

                    dyx = [(neigh[0] - y, neigh[1] - x) for neigh in neighs]
                    for dy, dx in dyx:

                         if dy < 0:
                              y_range = range(0, -ROWS, -1)
                         elif dy > 0:
                              y_range = range(0, ROWS)
                         else:
                              y_range = [0 for _ in range(ROWS)]

                         if dx < 0:
                              x_range = range(0, -COLUMNS, -1)
                         elif dx > 0:
                              x_range = range(0, COLUMNS)
                         else:
                              x_range = [0 for _ in range(COLUMNS)]

                         for cy, cx in zip(y_range, x_range):
                              pos = (y + cy, x + cx)

                              if pos != piece.position:
                                   if pos not in piece.attacking_positions:
                                        break

                                   if pos in board_getter.black_positions:
                                        break

                                   if pos in board_getter.white_positions:
                                        current_filtered.append(pos)
                                        break

                                   if pos in board_getter.free_positions:
                                        current_filtered.append(pos)

                         current_attacking_positions = current_filtered

          uids_with_valid_attacks[piece.uid] = current_attacking_positions


     return uids_with_valid_attacks


def filter_initial_moves(pieces: list[Piece],
                         uids_with_valid_attacks: dict[str, list[tuple[int, int]]],
                         board_getter: BoardGetter) \
        -> dict[str, list[tuple[int, int]]]:

     non_free_positions = board_getter.white_positions + board_getter.black_positions

     uids_with_initial_moves = dict()
     for piece in pieces:
          if piece.can_implement_pawn_moves:
               initial_moves = []
               initial_moves.extend(piece.moving_positions)

               if len(initial_moves) > 0:
                    if initial_moves[0] in non_free_positions:
                         initial_moves = []

                    if len(initial_moves) > 1:
                         if initial_moves[1] in non_free_positions:
                              initial_moves = [initial_moves[0]]

               initial_moves.extend(uids_with_valid_attacks[piece.uid])

          else:
               initial_moves = uids_with_valid_attacks[piece.uid]

          uids_with_initial_moves[piece.uid] = initial_moves

     return uids_with_initial_moves
