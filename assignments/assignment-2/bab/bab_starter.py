import picos as pic
from picos import RealVariable
from copy import deepcopy
from heapq import *
import heapq as hq
import numpy as np
import itertools
import math

counter = itertools.count()

y = RealVariable("y")


class BBTreeNode:
    """
    Creates and handles a BBTreeNode object that can branch and bound
    to determine the optimal result and corresponding best variable
    values.

    Attributes:
        vars (list of picos RealVariable objects): variables in the
            problem.
        constraints (list of constraints): list of problem constraints.
            ex: [z == x+y, -5*x+4*y <= 0, 6*x+2*y <= 17, x>=0, y>=0].
        objective (picos RealVariable object): variable that is being
            maximized.
        prob (picos Problem object): problem created by buildProblem
            using constraints, vars, and objective.
    """

    def __init__(self, vars=[], constraints=[], objective="", prob=None):
        """
        Initializes BBTreeNode.
        """
        self.vars = vars
        self.constraints = constraints
        self.objective = objective
        self.prob = prob

    def __deepcopy__(self):
        """
        Deepcopies the picos problem.
        This overrides the system's deepcopy method bc it doesn't work
        on classes by itself.

        Returns:
            (BBTreeNode object): copy of BBTreeNode.
        """
        newprob = pic.Problem.clone(self.prob)
        return BBTreeNode(self.vars, newprob.constraints, self.objective, newprob)

    def buildProblem(self):
        """
        Builds the initial Picos problem.

        Returns:
            self.prob (picos Problem object): problem created from
                constraints, objective, and vars.
        """
        prob = pic.Problem()
        prob.add_list_of_constraints(self.constraints)
        prob.set_objective("max", self.objective)
        self.prob = prob
        return self.prob

    def is_integral(self):
        """
        Checks if all variables (excluding the one we're maxing) are
        integers.

        Returns:
            (bool): returns True if all variables (excluding the one
                we're maximizing) are integers, otherwise False.
        """
        for v in self.vars[:-1]:
            if v.value == None or abs(round(v.value) - float(v.value)) > 1e-4:
                return False
        return True

    def branch_floor(self, branch_var):
        """
        Makes a child where xi <= floor(xi).

        Args:
            branch_var (float): variable to branch on.

        Returns:
            n1 (BBTreeNode object): child where xi <= floor(xi).
        """
        n1 = self.__deepcopy__()
        # add in the new binary constraint
        n1.prob.add_constraint(branch_var <= math.floor(branch_var.value))
        return n1

    def branch_ceil(self, branch_var):
        """
        Makes a child where xi >= ceiling(xi).

        Args:
            branch_var (float): variable to branch on.

        Returns:
            n2 (BBTreeNode object): child where xi >= ceiling(xi).
        """
        n2 = self.__deepcopy__()
        # add in the new binary constraint
        n2.prob.add_constraint(branch_var >= math.ceil(branch_var.value))
        return n2

    def bbsolve(self):
        """
        Uses the branch and bound method to solve an integer program.

        Returns:
            bestres (float): value of the maximized objective function.
            bestnode_vars (list of floats): list of variables that
                create bestres.
        """
        # these lines build up the initial problem and add it to a heap
        root = self
        res = root.buildProblem().solve(solver="cvxopt")
        heap = [(res, next(counter), root)]
        # set bestres to an arbitrary small initial best objective value
        bestres = -1e20
        # initialize bestnode_vars to the root vars
        bestnode_vars = root.vars
        # TODO:

        # Create an iteration variable less than the length of the heap
        iterator = 0
        while iterator < len(heap):
            maximum_tolerance = 0

            # Check if the constraints are valid for finding an IP solution
            if heap[iterator][2] not in ["infeasible", "unbounded"]:

                # Adds the maximum calculate values for constraints and
                # objective function if the numbers are integers
                if heap[iterator][2].is_integral():
                    if bestnode_vars[-1] < root.vars[-1]:
                        bestres = res
                        bestnode_vars = root.vars

                # If the numbers aren't integers, then B&B must be applied
                else:
                    # Calculates the variance from the closest integer of a variable
                    for i in range(len(heap[iterator][2].vars) - 1):
                        tolerance = heap[iterator][2].vars[i] - round(
                            heap[iterator][2].vars[i]
                        )
                        tolerance = abs(float(tolerance))
                        # The variable with the highest variance is selected where the ceiling
                        # and floor constraints will be applied
                        if maximum_tolerance < tolerance:
                            maximum_tolerance = tolerance
                            selection_variable = i

                    # Create the floor restriction for the lower branch
                    lower_branch = heap[iterator][2].branch_floor(
                        root.vars[selection_variable]
                    )

                    # Checks if PICOS is able to calculate an optimal solution
                    try:
                        lower_branch.prob.solve()

                    # Outputs a pruned branch that does not need to be explored further
                    except RuntimeError:
                        print("Pruned Solution Found")

                    # Adds the new and previous subjective constraints in the heap
                    else:
                        heappush(heap, (res, next(counter), lower_branch))

                    upper_branch = heap[iterator][2].branch_ceil(
                        root.vars[selection_variable]
                    )

                    # Checks if PICOS is able to calculate an optimal solution
                    try:
                        upper_branch.prob.solve()

                    # Outputs a pruned branch that does not need to be explored further
                    except RuntimeError:
                        print("Pruned Solution Found")

                    # Adds the new and previous subjective constraints in the heap
                    else:
                        heappush(heap, (res, next(counter), upper_branch))

                    # Reset the selection variable
                    selection_variable = None

            # Change the iteration
            iterator += 1

        # Return the optimal integer solutions once the loop has finished
        return bestres, bestnode_vars
