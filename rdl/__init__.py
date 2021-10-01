from pathlib import Path


PATH = Path(__file__).parent

# define the PATH to the folder to write the results in
PATH_SPEC = PATH / "path_spec.txt"

## specified
if PATH_SPEC.exists():
    print(f"File {PATH_SPEC.name} was found in\n{PATH_SPEC.parent}\n")
    with open(PATH_SPEC, "r") as f:
        PATH_RESULTS = Path(f.readline()[:-1])
        PATH_DATA = Path(f.readline()[:-1])
        PATH_MODELS = Path(f.readline()[:-1])

    print("Use the following specified locations for")
    print(f"  results: {PATH_RESULTS}")
    print(f"  data:    {PATH_DATA}")
    print(f"  models:  {PATH_MODELS}\n")

## default
else:
    print(f"File {PATH_SPEC.name} wasn't found in {PATH_SPEC.parent}\n")
    PATH_RESULTS = PATH.parent / "results"
    PATH_DATA = PATH.parent / "data"
    PATH_MODELS = PATH.parent / "models"

    print("Use the following default locations for")
    print(f"  results: {PATH_RESULTS}")
    print(f"  data:    {PATH_DATA}")
    print(f"  models:  {PATH_MODELS}")

    print("NOTE: you can specify your own locations by creating a file rdl/path_spec.txt consisting of a three lines with the paths to the desired folders (order must be results, data, models)\n")


from . import opt

from . import dataset
from . import utils

from . import dbqp
from . import classifier

from . import io
from . import plot
