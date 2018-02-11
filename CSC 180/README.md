# Hiya

Program to play a game designed by Scott Gordon @ CSU Sacramento.

********  TWO PLAYER GAMES - "Hi-YA!" ********

"Hi-YA!" is a brand new board game designed just for Fall 2017 CSc-180.
It has a sort of "martial arts" theme (i.e., Kung Fu or Karate), where the pieces
"fight", rather than try to capture each other.  And, as some of us know from watching
old Bruce Lee movies, a Karate chop is frequently accompanied by yelling "Hi-YA!".  

You will each write a program that can play "Hi-YA!" against a human opponent.
The game and the program requirements are detailed below.


THE GAME

"Hi-YA!" is a chess-like game in which each player takes turns moving one of
his/her pieces, and tries to knock out the opponent's King.  It is played on
a 7x8 board, and each side starts with a set of pieces consisting of 1 King,
3 Ninjas, 3 Samurai, 3 mini-Ninjas, and 3 mini-Samurai.  The different pieces
have rules about how they can move, which is described later.

Players alternate moves, moving one of their own pieces each turn.
When it is your turn, you MUST make a move -- you cannot pass.
If it is your turn and you have no legal moves, you lose.

A player wins either when he/she knocks out the opponents' King, or if it is
the opponent's turn but the opponent has no legal moves.

The initial position is:



   K,J,S,j,s mean King, Ninja, Samurai, mini-Ninja, and mini-Samurai, respectively.
      Those shown in RED belong to the computer and are moving DOWN the board.
      Those shown in WHITE belong to the human and are moving UP the board.

Unlike regular chess, the pieces in "Hi-YA!" may ONLY move onto empty squares.
They may NEVER move onto any square occupied by another piece.
Rather than "capturing" (as in regular chess), pieces in "Hi-YA!" attack each
other by moving onto the square directly IN FRONT OF an opposing piece.
For example, in the above picture, the Human could move his Ninja at E2 to B5,
and thus attack (or "fight") the computer's mini-Samurai at B6.  It is customary
in this game to announce "HiYa!!" when attacking an opponent's piece, and your
program is expected to do that when appropriate.

When a piece attacks an opposing piece, the opposing piece is either demoted
(becomes a weaker piece), or is removed from the board, depending on what type
of piece it is.  In particular:

- if attacked, a Ninja becomes a mini-Ninja.
- if attacked, a Samurai becomes a mini-Samurai.
- if attacked, a mini-Ninja is removed.
- if attacked, a mini-Samurai is removed.
- if attacked, a King is removed and that player loses.

The different pieces move as follows:

KING:
-- cannot move at all, ever.
-- whoever attacks the king wins!

NINJA:
-- moves roughly the same as the "bishop" in regular chess.
    That is, in a diagonal line any number of squares.
-- CANNOT jump over other pieces.
    Once it bumps into a piece (or the side of the board) that is as far as it can go.
-- a ninja can move in the forward diagonal direction.
    It can only move in a backwards diagonal direction if that move is an attack.
-- note that a ninja moves diagonally, but attacks by landing on the square directly
    below [not diagonal] (or above, in the case of the computer moving) an opposing piece.

SAMURAI:
-- moves roughly the same as a "rook" in regular chess.
    That is, in a horizontal or forward direction any number of squares.
-- never moves backwards.
-- CANNOT jump over other pieces.
    Once it bumps into a piece (or the side of the board) that is as far as it can go.
-- a samurai can move or attack in the forward direction.
    It can only move in the sideways direction if that move is a attack.

MINI-NINJA
-- moves the same as a NINJA, except that it can only move a distance of one square.
    That is, 1 square diagonally forward,
    or 1 square diagonally backwards if that move is an attack.
-- note that a mini-ninja moves diagonally, but attacks by landing on the square directly
    below [not diagonal] (or above, in the case of the computer moving) an opposing piece.

MINI-SAMURAI
-- moves the same as a SAMURAI , except that it can only move a distance of one square.
    That is, 1 square directly forward,
    or 1 square sideways if that move is an attack.



OTHER DETAILS - 

-- moving is compulsory.  That is, a player cannot "pass".
-- if you move in front of one of your own pieces, you are not attacking it.
-- unlike chess, there is no such thing as "check" or "checkmate".  Winning
     is by actually attacking the KING (or the opponent having no move).
-- unlike chess, there are no pawns, knights, or queens.
-- unlike chess, inability to move isn't a stalemate draw - it is a LOSS.
-- unlike Go (Weiqi), players don't place pieces on the board.
    The pieces are on the board at the beginning, and are moved.
-- NINJAs and SAMURAIs are captured by first being attacked and demoted to
    mini-NINJAs and mini-SAMURAIs, and then later removed in a future attack.
-- an attack happens only when someone MOVES a piece in front of another piece.
    For example, if you move a SAMURAI to attack your opponent's NINJA,
    your opponent's NINJA is demoted to a mini-NINJA and it is then your
    opponent's turn.  Your opponent's mini-NINJA is NOT then attacking your
    SAMURAI just because it is in front of it.  Attacking only happens when
    someone MOVES a piece in front of another piece.


THE PROGRAM

Your program is one player, and will attempt to defeat the human operator.
For full credit, it must fulfill the following requirements:

1. It first asks the human whether he/she wants to move first or second.

2. The current position is displayed on the screen before and after each
   move, with the axes labeled as in the following example:



   In the above example, colors are used to differentiate the human's pieces
   from the computer's pieces.  You are encouraged to consider alternative ways
   to display the board if you find a way that is easier to see (or more fun)
   with different characters for the pieces.  If you want to make a graphical
   user interface, that's even better (although you won't get extra credit for it).
   But whatever you do, the axes MUST be labeled, with row numbers and column
   letters, EXACTLY in the above manner. 

   (If, like me, you use simple "ascii" output to System.out, as shown above, you
    will probably find it a LOT easier to see the position if you use different
    colors for the human's pieces versus the computer's pieces.  This is actually
    very easy to do in C++ with ansi escape sequences, and your instructor will
    post the relevant code.  You are encouraged to add it to your program's display,
    although it isn't required.)

3. The human enters moves using the notation above, for the FROM square
   followed by the TO square.  For instance, in the position above, the
   human could use the SAMURAI on B2 to attack the computer's NINJA on B7
   and the move would be entered as follows: B2B7 (no spaces).  Of course,
   there are many other possible legal moves, such as B2B3, F2E3, or F2D4.
   You will probably find it more convenient if your program accepts both
   upper and lower case letters, but that isn't required.

4. After each move is made, the display should reflect how the board has changed.
   For example, if in the above diagram the human plays B2B6, the opponent's 
   NINJA would be "demoted" to a mini-NINJA, and the board should then be
   displayed as follows:



5. When your program makes a move, it must print out the move using the same
   notation described in #3 above, AND display the new position.  When printing out
   the computer's move, don't "flip" the axes - for example, in the above position,
   if the computer moves its mini-NINJA on the upper right, the move should be displayed
   as G6F5, that is, using exactly the same notation and the same axis labels.

6. If your program makes a move that attacks one of the human's pieces, it
   must also print out the words "Hi-YA!".  It shouldn't print those words if
   the move isn't an attacking move.  For example, an appropriate way to display
   the move described in #5 (above), would be:     G5F5 Hi-YA!

7. The program must detect when the game has ended, i.e. when a KING has
   has been attacked, or if a player has no legal moves and has thus lost.
   It should correctly announce when it has won, and should also correctly
   announce when the human has won.
  
8. The program must never play an illegal move.

9. The program should detect when an illegal move has been entered
   and require the human to re-enter a valid move.

10. At each turn, the program must make its move in 5 seconds or less.

11. The program should be sufficiently bug-free to play through an entire
   game without ever crashing.  Memory-leaks can cause these kinds
   of programs to fail late in the game, so test your program on several
   complete games, to be sure it is reliable through a whole game.

12. Your program must run on a CAMPUS workstation.  For the tournament,
   you MUST run on a campus workstation, not your own.  It could be one of
   the computers in 5029, or by connecting into one of the ECS servers.
   This is to make sure that everyone in the class has equal access to
   computing power.  If you have unique requirements, such as a wanting to
   use a language that isn't currently on a campus workstation, please
   make arrangements with me well ahead of the due date.


STRATEGY AND OTHER REQUIREMENTS

1. To get FULL credit, your program must satisfy the requirements listed
   above, and also MUST correctly use minimax search, at least 5 plies deep,
   with alpha-beta pruning (correctly), to determine the computer's moves.
   Refrain from trying to find a simple closed-form solution to the game,
   as that would defeat the purpose of the assignment.

2. You can use any programming language you want.  Any compiled language
   that generates efficient code, such as C or C++ are good choices.
   You can also use Java, C#, Python, Ruby, Perl, BASIC, Smalltalk, Scheme,
   LISP, Clojure, Javascript, or other interpreted languages, but don't be
   surprised if your code searches 1 or 2 plies less than those written
   in C or C++.  If you want to use a language other than one of the ones
   mentioned here, please see me first.

3. You can decide whether you want to do a deep selective search, or a
   more shallow but exhaustive (full-width) search.  Most programs like
   this perform better with exhaustive search, because a selective search
   would require an excellent heuristic to choose which moves to discard
   -- otherwise it would risk failing to consider important possibilities.
   You are strongly encouraged to use full-width search as described in class
   and in your textbook.
