import random
from typing import Annotated

import numpy as np
from pydantic import ConfigDict, Field, validate_call


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def split_dataset(X: np.ndarray, y: np.ndarray, ratio: Annotated[float, Field(strict=True, ge=0, le=1)] = 0.85):
    """
    Split the dataset into train and test datasets.
    """
    N = len(X)
    indices = list(range(N))
    random.shuffle(indices)
    train_indices, test_indices = indices[: round(ratio * N)], indices[round(ratio * N):]
    return X[train_indices], y[train_indices], X[test_indices], y[test_indices]
