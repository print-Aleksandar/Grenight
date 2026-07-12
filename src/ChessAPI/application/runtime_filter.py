from domain.pieces import Piece
from domain.board_configs import ROWS, COLUMNS


def filter_valid_attacks(white_pieces: list[Piece],
                         black_pieces: list[Piece],
                         white_positions: list[tuple[int, int]],
                         black_positions: list[tuple[int, int]],
                         free_positions: list[tuple[int, int]]) \
        -> list[tuple[Piece, list[tuple[int, int]]]]:

     non_white_positions = black_positions + free_positions
     non_black_positions = white_positions + free_positions

     pieces_with_valid_attacks = []

     for piece in white_pieces:
          current_attacking_positions = piece.attacking_positions

          if not piece.is_moving_sequence:
               current_attacking_positions = [position for position in current_attacking_positions
                                              if position in non_white_positions]

          else:
               if piece.do_attacking_position_requires_enemy_on_it:
                    current_attacking_positions = [position for position in current_attacking_positions
                                                   if position in black_positions]
               else:
                    current_filtered = []

                    y, x = piece.position
                    neighs = [neigh for neigh in current_attacking_positions
                              if (abs(neigh[0] - y) == 0 or abs(neigh[0] - y) == 1)
                              and (abs(neigh[1] - x) == 0 or abs(neigh[1] - x) == 1)]

                    dyx = [(y - neigh[0], x - neigh[1]) for neigh in neighs]
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

                              if pos not in piece.attacking_positions:
                                   break

                              if pos in white_positions:
                                   break

                              if pos in black_positions:
                                   current_filtered.append(pos)
                                   break

                              if pos in free_positions:
                                   current_filtered.append(pos)

                         current_attacking_positions = current_filtered

          pieces_with_valid_attacks.append((piece, current_attacking_positions))

     for piece in black_pieces:
          current_attacking_positions = piece.attacking_positions

          if not piece.is_moving_sequence:
               current_attacking_positions = [position for position in current_attacking_positions
                                              if position in non_black_positions]

          else:
               if piece.do_attacking_position_requires_enemy_on_it:
                    current_attacking_positions = [position for position in current_attacking_positions
                                                   if position in white_positions]
               else:
                    current_filtered = []

                    y, x = piece.position
                    neighs = [neigh for neigh in current_attacking_positions
                              if (abs(neigh[0] - y) == 0 or abs(neigh[0] - y) == 1)
                              and (abs(neigh[1] - x) == 0 or abs(neigh[1] - x) == 1)]

                    dyx = [(y - neigh[0], x - neigh[1]) for neigh in neighs]
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

                              if pos not in piece.attacking_positions:
                                   break

                              if pos in black_positions:
                                   break

                              if pos in white_positions:
                                   current_filtered.append(pos)
                                   break

                              if pos in free_positions:
                                   current_filtered.append(pos)

                         current_attacking_positions = current_filtered

          pieces_with_valid_attacks.append((piece, current_attacking_positions))

     return pieces_with_valid_attacks


def filter_valid_moves(white_pieces: list[Piece],
                       black_pieces: list[Piece],
                       white_positions: list[tuple[int, int]],
                       black_positions: list[tuple[int, int]],
                       free_positions: list[tuple[int, int]]) \
        -> list[tuple[Piece, list[tuple[int, int]], list[tuple[int, int]]]]:

     non_free_positions = white_positions + black_positions

     pieces_with_valid_attacks = filter_valid_attacks(white_pieces=white_pieces,
                                                      black_pieces=black_pieces,
                                                      white_positions=white_positions,
                                                      black_positions=black_positions,
                                                      free_positions=free_positions)

     piece_attack_moves = []
     for piece, valid_attacks in pieces_with_valid_attacks:
          if piece.can_implement_pawn_moves:
               initial_moving_positions = []
               initial_moving_positions.extend(piece.moving_positions)

               if len(initial_moving_positions) > 0:
                    if initial_moving_positions[0] in non_free_positions:
                         initial_moving_positions = None

                    if len(initial_moving_positions) > 1:
                         if initial_moving_positions[1] in non_free_positions:
                              initial_moving_positions = [initial_moving_positions[0]]

               initial_moving_positions.extend(valid_attacks)

          else:
               initial_moving_positions = valid_attacks

          piece_attack_moves.append((piece, valid_attacks, initial_moving_positions))

     return piece_attack_moves
