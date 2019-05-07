class DisableCV:
    """Class to disable Cross Validation. Insted of splits this class
    yields training and testing indices that include all given data.

    Implementation source: https://stackoverflow.com/a/55326439
    """

    def __init__(self):
        self.n_splits = 1

    def split(self, X, y, groups=None):
        yield range(len(X)), range(len(y))

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits
