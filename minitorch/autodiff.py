from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """

    vals_plus = list(vals)
    vals_plus[arg] += epsilon / 2
    vals_minus = list(vals)
    vals_minus[arg] -= epsilon / 2
    return (f(*vals_plus) - f(*vals_minus)) / epsilon

variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    visited = dict()
    top_sort_vars = []
 
    def topsort(root: Variable) -> None:

        visited[root.unique_id] = True

        if not root.is_leaf():
            for p in root.parents:
                if visited.get(p.unique_id) == None and p.is_constant() == False:
                    topsort(p)

        top_sort_vars.insert(0,root)

    
    topsort(variable)
    return top_sort_vars


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    topsortorder = topological_sort(variable)
    derivs = dict()
    derivs[variable.unique_id] = deriv

    for var in topsortorder:
        if not var.is_leaf():
            deriv = derivs.get(var.unique_id, 0)
            deriv = var.chain_rule(deriv)
            for input_var, d_output in deriv:
                if input_var.is_leaf():
                    input_var.accumulate_derivative(d_output)
                else:
                    derivs[input_var.unique_id] = derivs.get(input_var.unique_id, 0) + d_output



@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
