# Tetriling with missing pieces problem posed by Dr. Rojas
# Written by Chris Vail (CID 01387491) for DE2 COMP2 module "taught" by Nicolas Rojas

# 3 part solvers used to improve accuracy in a reasonable time
#   DLX - Perfect solver based on algorithm X
#   Greedy - Locally optimal shape selection without backtracking
#   GreedyMulti - Use Greedy to solve target in multiple orientations

from collections import deque   # Doubled ended queue for stacks and queues
import multiprocessing as mp    # Used to parallelise tasks
from time import perf_counter   # Keep track of slow DLX solves

from utils import generate_target

class SubSolver:

    """  
    Parent class to the various Solver classes (DLX, MultiGreedy, Greedy)
    Defines the interface required by Solver classes as well as common methods

    Note: Not implemented as an Abstract Base Class so Solver classes that don't
          implement the required methods will throw a NotImplementedError 
          during runtime if the required methods aren't overridden
    """

    # Stores the shapes that use the relative coordinates in order of most
    # important to least important. 
    # Used for the get_options method
    shape_coordinate_sets = (((1, 0), {4,5,7,8,10,11,12,13,14,16,17,19}), 
                             ((0, 1), {6,7,9,10,15,16,18}), 
                             ((1, 1), {6,11,12,13,15,17,18}), 
                             ((1, -1),{5,13,14,16,19}), 
                             ((2, 0), {4,8,10,12,14}), 
                             ((0, 2), {7,9,15}), 
                             ((1, 2), {9,11,18}), 
                             ((2, 1), {4,6,17}),
                             ((2, -1),{8,19}), 
                             ((1, -2),{5}))

    # Stores mapping of shapes to the other coordinates relative to the 
    # top-left most block
    shapes = {4 :((1, 0),  (2, 0),  (2, 1)),
              5 :((1, -2), (1, -1), (1, 0)),
              6 :((0, 1),  (1, 1),  (2, 1)),      
              7 :((0, 1),  (0, 2),  (1, 0)),
              8 :((1, 0),  (2, -1), (2, 0)),
              9 :((0, 1),  (0, 2),  (1, 2)),      
              10:((0, 1),  (1, 0),  (2, 0)),
              11:((1, 0),  (1, 1),  (1, 2)),
              12:((1, 0),  (1, 1),  (2, 0)),
              13:((1, -1), (1, 0),  (1, 1)),
              14:((1, -1), (1, 0),  (2, 0)),
              15:((0, 1),  (0, 2),  (1, 1)),       
              16:((0, 1),  (1,-1),  (1, 0)),      
              17:((1, 0),  (1, 1),  (2, 1)),
              18:((0, 1),  (1, 1),  (1, 2)),    
              19:((1, -1), (1, 0),  (2, -1))}


    def __init__(self, col_set):

        # No type checking as this is only called by the solve function.
        # Use a set for O(1) lookup and deleting
        self.col_set = col_set
        # Used a doubly linked list (as a stack) to avoid long append calls
        # of lists when extending the underlying array  
        self.solution = deque()

    def fill(self):
        """  
        This method should be used to generate a solution to the set of points given
        Solution should be stored in the self.solution variable
        Args:
            None
        Return:
            None
        """
        raise NotImplementedError()

    def build_solution(self):
        """  
        Given a solution this should yield (point, shape_id) tuples in the 
        order that they were placed down
        Args:
            None
        Return:
            Generator of point, shape_id tuples
        """
        raise NotImplementedError()

    def get_options(self, i, j):
        """  
        Given an i and j coordinate find the available shape options.
        Assumes the coordinate is for the top-left most block of the shape
        Only has to check each coordinate once so more efficient than a tree

        Checks each relative coordinate in order to remove the largest number
        of shapes first.

        If the coordinate is not available then it removes the set of shapes
        that require that block from the set of potentials. 
        
        If there are no options then return and empty set
        Otherwise return the left over shapes

        O(1) time complexity
        """
        # Generate the potential shapes 4 through 19
        potentials = {x for x in range(4, 20)}
        for (irel, jrel), vals in SubSolver.shape_coordinate_sets:
            if (i + irel, j + jrel) not in self.col_set:
                # Remove invalid shapes
                potentials -= vals
                if not potentials:
                    return set()

        return potentials


class Node:

    """  
    Node class for use in DLXSolver.
        > Node in the sparse matrix representation of the exact cover problem
        > Circular doubly linked list node with references to the column header
          as well as the shape id of the row
        > This should only be called from the DLXSolver class and as such doesn't 
          have many guard clauses. Use at your own risk.

    Methods:
        > Cover - cover this column of nodes
        > Uncover - uncover this column of nodes
        > get_other_row_nodes - get all nodes in the row except itself
        > get_other_row_nodes_rev - get all nodes in the row except itself
                                    but with a reversed order
        > insert_right - Insert a new node into this row
    """

    def __init__(self, column, shape_id, up=None, down=None):
        # References to column and row headers
        self.col = column
        self.shape_id = shape_id

        # Links in circular doubly linked list for columns
        # Each node is originally circularly linked to itself
        self.up = up
        self.down = down
        self.right = self
        self.left = self
    
    def cover(self):
        """  
        Covers a single block in the target and removes all related shapes

        Note: The order in which this occurs is of paramount importance
        """
        # Unlink column from headers
        self.col.cover()
        # Loop through all the nodes in the column
        for col_node in self.col.get_nodes():
            # Unlink all nodes in each row except the one in the selected column
            for node in col_node.get_other_row_nodes():
                node.up.down = node.down
                node.down.up = node.up
                # Keep track of the number of nodes under each column
                node.col.node_count -= 1

    def uncover(self):
        """  
        Reverse function of cover.
        Uncovers a single block in the target and all related nodes

        Again order of operations here is important
        """
        # Loop through all nodes in column
        for col_node in self.col.get_nodes_rev():
            # Link up all nodes in each row
            for node in col_node.get_other_row_nodes_rev():
                node.down.up = node
                node.up.down = node
                # Keep track of the number of nodes in each column
                node.col.node_count += 1

        # Link column header back up
        self.col.uncover()

    def get_other_row_nodes(self):
        """  
        Uses in cover and uncover to get all node except the one calling 
        the function
        """
        curr = self.right
        while curr is not self:
            yield curr
            curr = curr.right
    
    def get_other_row_nodes_rev(self):
        """  
        Same as get_other_row_nodes except in reverse order
        """
        curr = self.left
        while curr is not self:
            yield curr
            curr = curr.left
    
    def insert_right(self, node):
        """  
        Uses standard algorithm for inserting into a circular doubly linked list
        """
        self.left.right = node
        node.left = self.left
        self.left = node
        node.right = self

    def __iter__(self):
        """  
        Yields all nodes in the row.
        In other words provides all nodes needed to make a shape.
        """
        yield self
        curr = self.right
        while curr is not self:
            yield curr
            curr = curr.right


class NodeHeader:

    """  
    NodeHeader class for use in DLXSolver.
        > NodeHeader in the sparse matrix representation of the exact cover problem
          Represents a single available block in the target grid.
        > Circular doubly linked list node with references to the coordinate it 
          represents and the number of nodes under it.
        > This should only be called from the DLXSolver class and as such doesn't 
          have many guard clauses. Use at your own risk.

    Methods:
        > Cover - cover this column
        > Uncover - uncover this column
        > get_nodes - get all nodes in the column except itself
        > get_nodes_rev - get all nodes in the column except itself
                                    but with a reversed order
        > insert_right - Insert a new node into this row
        > insert_down - Insert a new node below this header
    """

    def __init__(self, col_id, node_count=0):
        # Coordinate of block
        self.col_id = col_id
        # Count of nodes for selection heuristic
        self.node_count = node_count
        # References for circular doubly linked list
        self.right = self
        self.left = self
        self.up = self
        self.down = self

    def insert_right(self, node):
        """ Inserts a node to the right of the header """
        self.left.right = node
        node.left = self.left
        self.left = node
        node.right = self

    def insert_down(self, node):
        """ Inserts a node below the header """
        self.up.down = node
        node.up = self.up
        self.up = node
        node.down = self
        self.node_count += 1

    def cover(self):
        """ Unlinks header from other headers in linked list """
        self.right.left = self.left
        self.left.right = self.right

    def uncover(self):
        """ Relinks header to other headers in linked list """
        self.right.left = self
        self.left.right = self
    
    def get_nodes(self):
        """ Iterates through nodes going down from the header """
        curr = self.down
        while curr is not self:
            yield curr
            curr = curr.down
    
    def get_nodes_rev(self):
        """ Reverse order iteration from get_nodes """
        curr = self.up
        while curr is not self:
            yield curr
            curr = curr.up

    def __iter__(self):
        # Loops through headers
        curr = self.right
        while curr is not self:
            yield curr
            curr = curr.right


class DLXSolver(SubSolver):
    
    """  
    Solver class that implements the dancing links algorithm popularised by
        Donald Knuth
    This algorithm is an obvious recursive solution to the exact cover problem
    It will only provide perfect solutions to the given set of available blocks

    Time complexity O(16**n) however in general much lower than this
    For efficient solving generally shouldn't be used for problems larger than 160
        blocks long. Increasing this will massively increase the run time
    """

    def __init__(self, col_set):
        """  
        Creates a DLXSolver object

        Args:
            col_set = a set of coordinates that need to be filled
        """
        super().__init__(col_set)
        # Header of the row headers. Will never be selected as only 64
        # rows can be underneath any column 
        self.head = NodeHeader(None, 65)
        # Provide a lookup for header node given the coordinate
        self.col_dict = {}

        # Generate the sparse matrix representation of the problem provided
        self._gen_cols()
        self._gen_sets()

    def fill(self, timeout=10):

        """  
        Perfectly solve the given problem using a DLX implementation and 
        stores the solution in self.solution.

        Must be called prior to calling build_solution.

        Uses an iterative implementation to avoid recursion limit errors 
        on large problems.

        Args:
            time_out (optional) - amount of time before DLX fails to continue
                                  solving. 0 gives unlimited time to solve.
        Returns:
            True if it successfully found 
        """

        t0 = perf_counter()
        # Ensure a block is placed on the first iteration
        back_track = False
        # Base case - No Blocks left to fill.
        while self.head.right is not self.head:
            
            # Check if DLX is taking too long and fail it
            if timeout and perf_counter() - t0 > timeout:
                return False

            # If undoing previous step
            if back_track:
                # Remove node from stack
                node = self.solution.pop()
                # Uncover related nodes
                for n in node.get_other_row_nodes_rev():
                        n.uncover()
                # Go to the next node
                node = node.down

                # If its a NodeHeader then its run out of options to backtrack
                if isinstance(node, NodeHeader):
                    back_track = True
                    # Uncover the column
                    node.down.uncover()
                    continue
                
                # Move on to next row in column
                else:
                    back_track = False

            else:
                # Choose column with least possible options - Prunes tree so
                # less repeated effort required
                col = min(self.head, key=lambda x: x.node_count)
                # If theres an impossible to reach block then backtrack
                if col.node_count == 0:
                    back_track = True
                    continue
                
                # Cover the column only once and select first node
                col.down.cover()
                node = col.down

            # Add node to solution stack
            self.solution.append(node)
            # Cover each node in the row to remove conflicting options
            for n in node.get_other_row_nodes():
                n.cover()
        return True

    def build_solution(self):
        """  
        Given a solution this yields (point, shape_id) tuples in the 
        order that they were placed down
        Args:
            None
        Return:
            Generator of point, shape_id tuples
        """
        for row in self.solution:
            for node in row:
                yield node.col.col_id, node.shape_id

    def _gen_cols(self):
        """  
        Uses the given set of coordinates to create this linked list of headers
        Should only be called during the class initialisation
        """
        for coord in self.col_set:
            curr = NodeHeader(coord)
            self.head.insert_right(curr)
            self.col_dict[coord] = curr

    def _gen_sets(self):
        """  
        Finds all possible shapes available to place and stores them as rows
        of nodes underneath the headers

        Should only be called during the class initialisation
        """
        for i, j in self.col_dict:
            for shape_id in self.get_options(i, j):
                # Make head for a row
                head = Node(self.col_dict[(i, j)], shape_id)
                head.col.insert_down(head)
                # Add in additional nodes
                for irel, jrel in self.shapes[shape_id]:
                    # Create a new node to insert
                    curr = Node(self.col_dict[(i + irel, j + jrel)], shape_id)
                    # Insert Node into circular doubly linked lists
                    head.insert_right(curr)
                    curr.col.insert_down(curr)


class GreedySolver(SubSolver):

    """  
    A Solver class that implements a greedy algorithm to imperfectly solve
    the given problem.
    Efficient for solving very large problems

    Time complexity of O(n) where n is the number of blocks to fill
    Average accuracy: 96.5% over a variety of sizes and densities
    """

    # Mapping of shapes to the relative coordinates of the blocks on the outside
    # of the shape. Used for selecting options
    shape_outside = {4:{(0, 1), (1, 1), (2, 2), (3, 1), (3, 0), (2, -1), (1, -1)},
                     5:{(0, 1), (1, 1), (2, 0), (2, -1), (2, -2), (1, -3)},
                     6:{(0, 2), (1, 2), (2, 2), (3, 1), (2, 0), (1, 0)},
                     7:{(0, 3), (1, 2), (1, 1), (2, 0), (1, -1)},
                     8:{(0, 1), (1, 1), (2, 1), (3, 0), (3, -1), (2, -2), (1, -1)},
                     9:{(0, 3), (1, 3), (2, 2), (1, 1), (1, 0)},
                     10:{(0, 2), (1, 1), (2, 1), (3, 0), (2, -1), (1, -1)},
                     11:{(0, 1), (0, 2), (1, 3), (2, 2), (2, 1), (2, 0), (1, -1)},
                     12:{(0, 1), (1, 2), (2, 1), (3, 0), (2, -1), (1, -1)},
                     13:{(0, 1), (1, 2), (2, 1), (2, 0), (2, -1), (1, -2)},
                     14:{(0, 1), (1, 1), (2, 1), (3, 0), (2, -1), (1, -2)},
                     15:{(0, 3), (1, 2), (2, 1), (1, 0)},
                     16:{(0, 2), (1, 1), (2, 0), (2, -1), (1, -2)},
                     17:{(0, 1), (1, 2), (2, 2), (3, 1), (2, 0), (1, -1)},
                     18:{(0, 2), (1, 3), (2, 2), (2, 1), (1, 0)},
                     19:{(0, 1), (1, 1), (2, 0), (3, -1), (2, -2), (1, -2)}}

    def __init__(self, coord_set):
        """  
        Creates a GreedySolver object

        Args:
            coord_set = a set of coordinates that need to be filled
        """
        super().__init__(coord_set)
        self.height_range, self.width_range = self.get_dimensions()
        # Best order to place shapes in using this shape placing order
        # Order gained from progressive testing
        # Up to 10% more accurate than randomly selecting shapes
        self.shape_selection_oder = (16, 5, 7, 15, 18, 13, 10, 14, 12, 17, 11, 19, 8, 4, 9, 6)
        # Number of blocks that were unable to be filled in
        self.unreachable = 0
   
    def get_dimensions(self):
        """  
        Get the coordinates of the bounding box for the given problem
        """
        height_range = [1000, 0]
        width_range = [1000, 0]
        for i, j in self.col_set:
            if i < height_range[0]: height_range[0] = i
            if i > height_range[1]: height_range[1] = i
            if j < width_range[0]: width_range[0] = j
            if j > width_range[1]: width_range[1] = j
        
        # Adjust ranges to include end points
        height_range[1] += 1
        width_range[1] += 1

        return height_range, width_range

    def fill(self):
        """  
        Fills the problem from top to bottom, left to right placing locally
            optimal shapes at each stage. If a piece can not be placed then 
            it is skipped over and ignored
        Solution is stored in the self.solution instance variable
        """
        # Loop though all potential coordinates
        for i in range(*self.height_range):
            for j in range(*self.width_range):
                # Continue if the square isn't available
                if (i, j) not in self.col_set: continue

                # Find all possible options then select the best
                options = self.get_options(i, j)
                if not options:
                    self.unreachable += 1
                    continue
                selected_option = self._select_option(i, j, options)

                # Place the selected option, remove covered squares 
                # and add to solution
                self.col_set.remove((i, j))
                self.solution.append((selected_option, (i, j)))
                for irel, jrel in GreedySolver.shapes[selected_option]:
                    self.col_set.remove((i + irel, j + jrel))
                    self.solution.append((selected_option, (i + irel, j + jrel)))

    def build_solution(self):
        """  
        Given a solution this yields (point, shape_id) tuples in the 
        order that they were placed down
        Args:
            None
        Return:
            Generator of point, shape_id tuples
        """
        for shape_id, coord in self.solution:
            yield coord, shape_id
        
    def _select_option(self, i, j, options):
        """  
        Finds the locally optimal shape to place given an i, j coordinate
            and a set of options from which to pick
        
        Should only be called by internal methods
        
        Time complexity of O(n) in number of options

        Args:
            i - i coordinate of position to place shape
            j - j coordinate of position to place shape
            options - set of options to choose from
        
        Returns:
            Shape_id of the locally selected piece
        """
        # If only 1 option then return it
        if len(options) == 1:
            return options.pop()
        
        # Initialise best
        best = None

        # Score each option in selection order
        for count, shape in enumerate(x for x in self.shape_selection_oder if x in options):
            # Initialise score
            score = 0
            # Squares that would be covered if the block was placed
            covered = set((i, j), )
            for irel, jrel in GreedySolver.shapes[shape]:
                covered.add((i + irel, j + jrel))

            # Blocks that have been seen already by flood_filling
            skip = set()
            # Go through surrounding blocks and score
            for irel, jrel in GreedySolver.shape_outside[shape]:
                if (i + irel, j + jrel) in skip: continue
                # Get the score for the block and update variables
                block_score, to_skip = self.score_block(i + irel, j + jrel, covered)
                score += block_score
                skip |= to_skip

            # Update best option
            if count == 0 or score > best[1]:
                best = (shape, score)

            # Returns early if it finds a better than average shape
            # Based on the results of scoring all options > 1,000,000 times
            # Greatly increases efficiency with minimal accuracy loss
            #
            # Average top score: 32.84779771650669
            # Average average score: 18.856394946704214
            # Average bottom score: 3.9737942582581534
            if score > 20: 
                return shape
                
        return best[0]

    def score_block(self, i, j, covered):
        """  
        Determines a score based on whether the shapes fits in well or if 
            it will leave unreachable blocks
        
        Args:
            i - i coordinate of the block to score
            j - j coordinate of the block to score
            covered - set of coordinates that would be covered if the shape
                      was placed
        
        Returns:
            A tuple: (score for the block, set of seen blocks)
        """
        # Improve score for fitting in
        if (i, j) not in self.col_set:
            return 10, set()

        # Blocks still to be flooded
        to_be_filled = deque(((i, j), ))
        # Blocks that have already been seen
        seen = set(((i, j),)) | covered
        block_count = 1

        # Iterative flood fill algorithm (BFS) to find connected blocks
        while to_be_filled:
            i, j = to_be_filled.popleft()
            # Test all block around the current one
            for irel, jrel in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                coord = (i + irel, j + jrel)
                if coord in self.col_set and coord not in seen:
                    seen.add(coord)
                    to_be_filled.append(coord)
                    block_count += 1

            # If greater than 4 block are found then a piece can be placed
            if block_count >= 4:
                return 0, seen

        # If less than 4 blocks found, decrement score for each unfillable
        # block
        return block_count*(-10), seen


class Transformer:

    """  
    Utility class that allows users to transform a set of coordinates into 
        the 8 different orientations possible without scaling, solve a 
        tetromino problem and then reorient the solution.
    """
    
    # Mapping of linear transforms and their inverse to the functions used 
    # to implement them
    map_funcs = {"10;01":{"map":lambda x: x, "unmap":lambda x: x},
                 "0-1;10":{"map":lambda x: (-x[1], x[0]), "unmap":lambda x: (x[1], -x[0])},
                 "-10;0-1":{"map":lambda x: (-x[0], -x[1]), "unmap":lambda x: (-x[0], -x[1])},
                 "01;-10":{"map":lambda x:(x[1], -x[0]), "unmap":lambda x: (-x[1], x[0])},
                 "-10;01":{"map":lambda x: (-x[0], x[1]), "unmap":lambda x: (-x[0], x[1])},
                 "10;0-1":{"map":lambda x: (x[0], -x[1]), "unmap":lambda x: (x[0], -x[1])},
                 "0-1;-10":{"map":lambda x: (-x[1], -x[0]), "unmap":lambda x: (-x[1], -x[0])},
                 "01;10":{"map":lambda x: (x[1], x[0]), "unmap":lambda x: (x[1], x[0])}}

    # Maps shapes found in a transform to their untransformed shape id
    shape_map = {'10;01': {4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19}, 
                 '0-1;10': {4: 7, 5: 4, 6: 5, 7: 6, 8: 11, 9: 8, 10: 9, 11: 10, 12: 15, 13: 12, 14: 13, 15: 14, 16: 17, 17: 16, 18: 19, 19: 18}, 
                 '-10;0-1': {4: 6, 5: 7, 6: 4, 7: 5, 8: 10, 9: 11, 10: 8, 11: 9, 12: 14, 13: 15, 14: 12, 15: 13, 16: 16, 17: 17, 18: 18, 19: 19}, 
                 '01;-10': {4: 5, 5: 6, 6: 7, 7: 4, 8: 9, 9: 10, 10: 11, 11: 8, 12: 13, 13: 14, 14: 15, 15: 12, 16: 17, 17: 16, 18: 19, 19: 18}, 
                 '-10;01': {4: 10, 5: 9, 6: 8, 7: 11, 8: 6, 9: 5, 10: 4, 11: 7, 12: 12, 13: 15, 14: 14, 15: 13, 16: 18, 17: 19, 18: 16, 19: 17}, 
                 '10;0-1': {4: 8, 5: 11, 6: 10, 7: 9, 8: 4, 9: 7, 10: 6, 11: 5, 12: 14, 13: 13, 14: 12, 15: 15, 16: 18, 17: 19, 18: 16, 19: 17}, 
                 '0-1;-10': {4: 11, 5: 10, 6: 9, 7: 8, 8: 7, 9: 6, 10: 5, 11: 4, 12: 13, 13: 12, 14: 15, 15: 14, 16: 19, 17: 18, 18: 17, 19: 16}, 
                 '01;10': {4: 9, 5: 8, 6: 11, 7: 10, 8: 5, 9: 4, 10: 7, 11: 6, 12: 15, 13: 14, 14: 13, 15: 12, 16: 19, 17: 18, 18: 17, 19: 16}}

    def get_all_set_transforms(self, coord_set):
        """  
        Yields all possible transforms of the coordinate set

        Args:
            coord_set - any iterable of coordinates to be transformed
        Returns:
            Generator that yields sets of transformed coordinates and 
            the transform used
        """
        for transform, mapping in Transformer.map_funcs.items():
            # map applies the mapping function to each coordinate in the set
            yield set(map(mapping["map"], coord_set)), transform

    def get_reverse_transform_func(self, transform):
        """  
        Get the function to reverse the transform provided

        Args:
            transform - transform to get the inverse of in the format given 
                        out by get_all_set_transforms
        Returns:
            Function object that takes a transformed coordinate and 
            returns the untransformed coordinate
        """
        return Transformer.map_funcs[transform]["unmap"]

    def map_shape(self, transform, shape_id):
        """  
        Get the untransformed shape id of the transformed shape

        Args:
            transform - transform to get the inverse of in the format given 
                        out by get_all_set_transforms
            shape_id - shape id from the transformed solution
        Returns:
            shape id of the untransformed shape        
        """
        return Transformer.shape_map[transform][shape_id]


class GreedyMultiSolver(SubSolver):

    """  
    A Solver class that is ideal for medium size targets < 100x100. These 
        targets suffer from being too small, leading to uncharacteristicly 
        low accuracy solves. This class solves the grid in 8 different 
        orientations and selects the best option to overcome this
    
    Time complexity of O(n) in number of blocks to fill
    """

    def __init__(self, coord_set):
        """  
        Creates a GreedyMultiSolver object

        Args:
            coord_set = a set of coordinates that need to be filled
        """
        super().__init__(coord_set)
        # Stores (number of unreachable squares, solver object, transform)
        self.best_solution = (float("inf"), None, None)
        self.transformer = Transformer()

    def fill(self):
        """  
        Fills the problem with shapes in various different orientations
            in order to get a reasonable accuracy of solve.
        Solution is stored in the self.solution instance variable
        """
        # Get each transformed set of points
        for col_set, transform in self.transformer.get_all_set_transforms(self.col_set):
            # Use a standard greedy solver to solve each one
            solver = GreedySolver(col_set)
            solver.fill()
            # Select the best one
            if solver.unreachable < self.best_solution[0]:
                self.best_solution = (solver.unreachable, solver, transform)       

    def build_solution(self):
        """  
        Given a solution this yields (point, shape_id) tuples in the 
        order that they were placed down
        Args:
            None
        Return:
            Generator of point, shape_id tuples
        """
        transform_func = self.transformer.get_reverse_transform_func(self.best_solution[2])

        for coord, shape_id in self.best_solution[1].build_solution():
            # Untransform the coordinate and shape id
            yield transform_func(coord), self.transformer.map_shape(self.best_solution[2], shape_id)


class ProblemSplitter:

    """  
    A utility class that breaks a target area into independant subproblems.
    Instances to be used as a generator function yielding the coordinates of 
        each available square in each subproblem and the length of the problem
    """

    def __init__(self, target):
        """  
        Prepare target for subproblem finding

        Args:
            target - Target is a 2d array of boolean values where 1's are to 
                     be filled and 0's are to be ignored
        """
        width = len(target[0]) + 2
        self.target = [[0 for _ in range(width)]]
        for row in target:
            self.target.append([0] + row + [0])
        self.target.append([0 for _ in range(width)])

    def __iter__(self):
        """  
        Iterate though each subproblem, find it and then yield it
        """
        for i, row in enumerate(self.target):
            for j, item in enumerate(row):
                # Skip 0's
                if not item: continue

                yield self._flood_fill(i, j)

    def _flood_fill(self, i, j):
        """  
        BFT of coordinates surrounding the given coordinate

        Args:
            i, j - coordinates of the block known to contain a 1
        Returns:
            Tuple containing the set of coordinates found and the size of the 
            subproblem
        """
        coords = set(((i, j), ))
        shape = set(((i - 1, j - 1), ))
        size = 0
        while coords:
            i, j = coords.pop()
            for irel, jrel in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                if self.target[i + irel][j + jrel]:
                    size += 1
                    shape.add((i + irel - 1, j + jrel - 1))
                    coords.add((i + irel, j + jrel))
                    self.target[i + irel][j + jrel] = 0

        return shape, size


def worker(solver_cls, target):
    """  
    Multiprocessing worker function that takes in a solver class and a 
    subproblem and returns a solution

    Args:
        solver_cls - A child of SubSolver to be used to solve the problem
        cols - a set of coordinates that define the area to be filled
    Returns:
        Generator object for the solution to the problem
    """
    # Check to ensure correct class passed
    if not issubclass(solver_cls, SubSolver):
        raise TypeError("Solver must be a subclass of SubSolver to be used")
    # Initialise then solve
    solver = solver_cls(target)
    if isinstance(solver, DLXSolver):
        if not solver.fill(timeout=15):
            solver = GreedyMultiSolver(target)
            solver.fill()
    else:
        solver.fill()
    return [x for x in solver.build_solution()]


def solve(target):
    """  
    Solve the tetromino tiling problem given by target.
    Function acts as a factory method -> distributing the solutions and 
        interpreting the results 
    Note: Uses multiprocessing that is not entirely stable across OS's

    Args:
        Target - 2d array of 1's for block to fill and 0's for blocks to ignore
    Returns:
        2d array filled with tuples containing the shape placed and the placing 
        order 
    """
    # Store basic target properties
    target = target
    height = len(target)
    width = len(target[0])
    size = len(target) * len(target[0])
    density = sum([sum(x) for x in target])/(size)

    # Initialise multiprocessing queue of tasks
    tasks = deque()
    # Split problem into independant subproblems and create multiprocessable tasks
    splitter = ProblemSplitter(target)
    for subproblem, length in splitter:
        # Perfectly solve smaller subproblems or low density problems
        if density < 0.38 or length < 257:
            solver = DLXSolver
        # Ensure higher accuracy for smaller problems
        elif size < 100*100 + 1:
            solver = GreedyMultiSolver
        # Quick imperfect solver for all large subproblems
        else:
            solver = GreedySolver

        tasks.append((solver, subproblem))
        
    # Use all cores to process the tasks found above
    # Uses the same number as threads as cores as the problem is CPU bound
    # mp.Pool manages processes and aggregates the results
    # starmap gives the parallised function the tuple of arguments supplied
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(worker, tasks)
    # Uncomment these lines if multiprocessing is not working
    """ results = []
    for task in tasks:
        results.append(worker(*task)) """

    # Initialise solution
    sol = [[(0, 0) for __ in range(width)] for _ in range(height)]
    # Build solution from the results
    level = 0
    for solution in results:
        for (i, j), shape_id in solution:
            # level//4 + 1 so every 4 blocks get the same level and it starts at 1
            sol[i][j] = (shape_id, level//4 + 1)
            level += 1

    return sol


def solve_iterable(target):
    # Store basic target properties
    target = target
    height = len(target)
    width = len(target[0])
    size = len(target) * len(target[0])
    density = sum([sum(x) for x in target])/(size)

    # Initialise multiprocessing queue of tasks
    tasks = deque()
    # Split problem into independant subproblems and create multiprocessable tasks
    splitter = ProblemSplitter(target)
    for subproblem, length in splitter:
        # Perfectly solve smaller subproblems or low density problems
        if density < 0.38 or length < 512:
            solver = DLXSolver
        # Ensure higher accuracy for smaller problems
        elif size < 100*100 + 1:
            print("Using greedy solver")
            solver = GreedyMultiSolver
        # Quick imperfect solver for all large subproblems
        else:
            solver = GreedySolver

        tasks.append((solver, subproblem))
        
    # Use all cores to process the tasks found above
    # Uses the same number as threads as cores as the problem is CPU bound
    # mp.Pool manages processes and aggregates the results
    # starmap gives the parallised function the tuple of arguments supplied

    """ with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(worker, tasks) """
    
    results = []
    for job in tasks:
        results.append(worker(*job))
    
    # Uncomment these lines if multiprocessing is not working
    """ results = []
    for task in tasks:
        results.append(worker(*task)) """

    # Initialise solution
    sol = [[(item*80, 0) for item in row] for row in target]
    #sol = [[(0, 0) for item in row] for row in target]
    # Build solution from the results
    level = 0
    for solution in results:
        for (i, j), shape_id in solution:
            # level//4 + 1 so every 4 blocks get the same level and it starts at 1
            sol[i][j] = (shape_id, level//4 + 1)
            level += 1
            if level%4 == 0:
                yield sol

# PEP8 states that functions should never start with a capital letter however
# the function required by the brief is Tetris so this line gets around that
Tetris = solve

if __name__ == "__main__":
    nx = 37 
    ny = 11
    density = 0.5
    ignore = {1,2,3}
    target = generate_target(nx, ny, density, ignore)[0]
    solution = solve(target)

    show_solution(target, solution)