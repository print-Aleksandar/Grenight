import pytest
from domain.configs import ROWS, COLUMNS
from domain.pieces import Piece, King, Bishop
from domain.board_initialization import create_initial_board


class TestBoardInitialization:

    def test_does_initialization_has_correct_number_of_pieces(self):
        """
        on initial chess board four rows have piece
        on each square with the width of columns per row
        """

        assert len(create_initial_board()) == COLUMNS * 4


    def test_does_contain_white_king(self):
        assert any(p for p in create_initial_board() if type(p) == King and p.is_white)

    def test_does_contain_black_king(self):
        assert any(p for p in create_initial_board() if type(p) == King and not p.is_white)

    def test_do_white_pieces_are_in_first_two_rows(self):
        pieces = create_initial_board()
        white_pieces = [p for p in pieces if p.is_white]

        assert all(p.position[0] in (0, 1) for p in white_pieces)

    def test_do_black_pieces_are_in_last_two_rows(self):
        pieces = create_initial_board()
        black = [p for p in pieces if not p.is_white]

        assert all(p.position[0] in (ROWS - 1, ROWS - 2) for p in black)

    def test_do_kings_are_in_same_column(self):
        pieces = create_initial_board()
        kings = [p for p in pieces if type(p) == King]

        assert all(p.position[1] == kings[0].position[1] for p in kings)