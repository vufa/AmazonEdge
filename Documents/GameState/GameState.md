 in `AmazonEdge/amazon.py GameState`:

## Value in init

* `board`: a `10x10` array, init with `EMPTY`
* `size`: the board size, init is `10`
* `current_player`: the current player, init is also the first, is black
* `num_black_prisoners`: prisoners of black(cannot move because surrounded by barriers)
* `num_black_prisoners`: same as above
* `is_end_of_game`: used to judge whether the game is end
* `liberty_sets`: is a `10(10x3)x10` 2D set() type array, used to directly optimize update function
* `liberty_counts`: a `10x10` int type array, liberty of each chessman on board, init with `-1`


## Function

* `_create_neighbors_cache`: