from typing import List
from numpy.random import uniform
class GenerateDatasets:

    OPERATORS = ["+", "-", "*", "/"]
    def __init__(self, equations: List[List[str]] = None, X_values: List[str] = None):
        self.equations = equations
        self.X_values = X_values
    
    def generate_xy_dataset(self) -> float:
        
        for x in self.X_values:
            for equation in self.equations:
                expression = [] 
                for token in equation:
                    if token in GenerateDatasets.OPERATORS:
                        expression.append(token)
                    elif token == "x":
                        expression.append(f"{x}")
                    else:
                        expression.append(token)
                
                #print(f"Expression as list is: {expression}")
                expression = "".join(expression)
                #print(f"Expression as string is: {expression}")
                value = eval(expression)
                print(f"Value of expression {''.join(equation)} for X = {x} is {value}")
        return 0

if __name__ == "__main__":
    list_of_equations = [["3","*","x","*","x","+","2","*","x","+","1"], ["5","*","x","-","3"]]
    X_values = uniform(-1, 1, 5)
    dataset_generator = GenerateDatasets(equations= list_of_equations, X_values=X_values)
    dataset_generator.generate_xy_dataset()





