from random_quadratics import RandomQuadratic
from generate_datasets import GenerateDatasets
from equations_dataset import EquationsDataset
from numpy.random import uniform
import argparse


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


def main():

    args = parse_arguments()
    equations = []

    for _ in range(args.number_of_equations):
        equation = RandomQuadratic(
            args.lower_bound, args.upper_bound, args.round_digits
        )
        equations.append(equation())

    X_values = uniform(-1, 1, 5).astype(dtype="float64", copy=False)
    root_dir = "./datasets"
    dataset_generator = GenerateDatasets(equations=equations, X_values=X_values)
    datesets_dict = dataset_generator.generate_xy_datasets()
    GenerateDatasets.write_dataset_to_file(datesets_dict, root_dir=root_dir)


if __name__ == "__main__":
    main()
