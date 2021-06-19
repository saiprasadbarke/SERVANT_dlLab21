from typing import List, Union
import argparse
from pprint import pprint
from random import randint, choice, uniform
import re


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lower-bound",
        type=int,
        default=-5,
        help="lower bound for generating constants",
    )
    parser.add_argument(
        "--upper-bound",
        type=int,
        default=5,
        help="upper bound for generating constants",
    )
    parser.add_argument(
        "--round-digits",
        type=int,
        default=3,
        help="number of round digits for floating numbers",
    )
    parser.add_argument(
        "--number-of-equations",
        type=int,
        default=2,
        help="number of random quadratics to generate",
    )
    args = parser.parse_args()
    return args


class RandomQuadratic:
    """
    Create random quadratic equations in the form (+/-) a*x*x (+/-) b*x (+/-) c
    where a, b, c are randomly generated integers or floating points,
    the signs for the coefficients and constants is also randomized by choice,
    the resulting equations are then cleaned in the sense to remove redunant
    signs such a +- to -, -- to + and so on. This might be adapted such that
    the substitions will not be need, however at the moment we utilize the former.
    """

    def __init__(
        self, lower_bound: int = None, upper_bound: int = None, round_digits: int = None
    ):
        """
        Args:
            lower_bound: lower bound for generating constants
            upper_bound: upper_bound: upper bound for generating constants
            round_digits: number of round digits for floating numbers
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.round_digits = round_digits
        self
        self.a = self.generate_number()
        self.b = self.generate_number()
        self.c = self.generate_number()
        self.equation = None

    def generate_number(self) -> Union[int, float]:
        """
        Generate random numbers, here we use randint to generate random integers between the lower bound and upper bound and we use uniform for generate random floating
        points between the aforementioned range. The selection is randomized by choice function.

        Returns:
            Random number, either a integer or a floating point
        """
        return choice(
            [
                randint(self.lower_bound, self.upper_bound),
                round(uniform(self.lower_bound, self.upper_bound), self.round_digits),
            ]
        )

    def generate_equation(self):
        """
        Generates a quadratic equation as specified in the class description.
        """
        if self.a == 0:
            self.equation = str(
                choice(["+", "-"])
                + str(self.b)
                + "*x"
                + choice(["+", "-"])
                + str(self.c)
            )
        elif self.b == 0:
            self.equation = str(
                choice(["+", "-"])
                + str(self.a)
                + "*x*x"
                + choice(["+", "-"])
                + str(self.c)
            )
        else:
            self.equation = str(
                choice(["+", "-"])
                + str(self.a)
                + "*x*x"
                + choice(["+", "-"])
                + str(self.b)
                + "*x"
                + choice(["+", "-"])
                + str(self.c)
            )

    def __call__(self) -> List[str]:
        """
        Here we clean the generated equation to simplify redundant signs such as -- to +,
        we also split the equation to the format whereby we can use the equation to
        create (X,y) datset furtheron.

        Returns:
            Equation as a list of symbols, constants and operators
        """
        self.generate_equation()
        self.equation = re.sub(r"(?:\+-|\-\+)", "-", self.equation)
        self.equation = re.sub(r"(?:\--)", "+", self.equation)
        return re.split(r"([\+\-\*])", self.equation)[1:]


if __name__ == "__main__":
    args = parse_arguments()
    print("hello ?")
    pprint(vars(args))
    print("not now marge !")
    equations = []

    for _ in range(args.number_of_equations):
        equation = RandomQuadratic(
            args.lower_bound, args.upper_bound, args.round_digits
        )
        equations.append(equation())

    pprint(equations)
