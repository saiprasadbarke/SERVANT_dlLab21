from typing import List, Dict
from numpy.random import uniform
from json import dumps, dump


class GenerateDatasets:
    def __init__(self, equations: List[List[str]] = None, X_values: List[float] = None):
        self.equations = equations
        self.X_values = X_values

    def generate_xy_datasets(self) -> Dict[str, Dict[float, float]]:
        """
        This method calls generate_xy_dataset_for_equation for every equation and returns a dictionary with the key as the equation in string form and the value as the dictionary of X-Y values for the said equation.
        """
        dictionary_of_xy_values_for_equations = dict()
        for equation in self.equations:
            dictionary_of_xy_values_for_equations[
                f"{''.join(equation)}"
            ] = self.generate_xy_dataset_for_equation(equation)
        return dictionary_of_xy_values_for_equations

    def generate_xy_dataset_for_equation(
        self, equation: List[str]
    ) -> Dict[float, float]:
        """
        This method returns a dictionary of X-Y values for a given equation
        """
        dictionary_of_xy_values = dict()
        for x in self.X_values:
            expression = []
            for token in equation:
                if token == "x":
                    expression.append(f"{x}")
                else:
                    expression.append(token)
            expression = "".join(expression)
            value = eval(expression)
            dictionary_of_xy_values[x] = value
        return dictionary_of_xy_values

    @staticmethod
    def write_dataset_to_file(
        dictionary_of_xy_values_for_equations: Dict[str, Dict[float, float]]
    ):
        with open("datasets_equations.json", "w") as outfile:
            dump(dictionary_of_xy_values_for_equations, outfile, indent=4)


if __name__ == "__main__":
    list_of_equations = [
        ["3", "*", "x", "*", "x", "+", "2", "*", "x", "+", "1"],
        ["5", "*", "x", "-", "3"],
    ]
    X_values = uniform(-1, 1, 5)
    dataset_generator = GenerateDatasets(equations=list_of_equations, X_values=X_values)
    datesets_dict = dataset_generator.generate_xy_datasets()
    # dumps makes the float X values in keys into strings
    print(dumps(datesets_dict, indent=4))
    GenerateDatasets.write_dataset_to_file(datesets_dict)
