# FOR-RoboMarkt

This repository contains the delivered implementation for the _Foundations of Operations Research_ course's project at Politecnico di Milano, A.Y. 2021/2022.

Devs: [Ottavia Belotti](https://github.com/OttaviaBelotti), [Alessio Braccini](https://github.com/AlessioBraccini), [Martin Bronzo](https://github.com/MartinBronzo)


## Specifications

The project consits in solving an optimization problem: given _n_ village locations, **minimize the building cost** of stores while still serving all of the locations. A village is served if there is at least one open store within an acceptable distance range.

At the same time, refurbishing of the built shops must be planned out, optimizing again: restock all the stores while **minimizing the total cost** to do so. The total cost is comprehensive of both fixed costs and variable costs.

> More details about the problem and all the constraints can be found in [Specifications](https://github.com/OttaviaBelotti/FOR-RoboMarkt/blob/main/Specifications.pdf).

## Solution
The problem has been tackled by dividing it into two sub-problems: finding all the locations where to build the stores first, then planning out the refurbishing process for the chosen and built stores. The final code can be found in [model.py](https://github.com/OttaviaBelotti/FOR-RoboMarkt/blob/main/model.py).

> More about the solution in [RoboMarkt report](https://github.com/OttaviaBelotti/FOR-RoboMarkt/blob/main/RoboMarkt%20report.pdf).

### 1. Build Stores
The optimal solution has been found through formal modelization and solved thanks to MIP Python library. 
### 2. Restock Stores
This part has been carried out with an heuristic approach, beign notably a NP-hard problem. It has been solved by merging both _Neighborhood algorithm_ and _Clarke-Wright algorithm_. The one that performs better at run-time given the input will return the final solution. Read the report for more details.

## Datasets
You can find the public datasets used for testing the code in the [datasets](https://github.com/OttaviaBelotti/FOR-RoboMarkt/tree/main/datasets) folder.

## Papers and Tools
* [MIP Python library](https://www.python-mip.com/)
* [Single-Depot VRP and Clarke-Wright savings algorithm](https://web.mit.edu/urban_or_book/www/book/chapter6/6.4.12.html)
