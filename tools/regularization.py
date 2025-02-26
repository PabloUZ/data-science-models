import copy


class Regularization:
    """
    Applies different types of regularization techniques to a model.

    Regularization helps prevent overfitting by penalizing large or unnecessary weights.
    This class provides implementations for L0, L1, and L2 regularization norms.

    Attributes:
        model (object): The model containing weights to be regularized.
                        It must have an attribute `w`, which is a list or array of weights.
        weight_decay (float): The regularization strength (default is 1).
    """
    def __init__(self, model, weight_decay = 1):
        """
        Initializes the Regularization class.

        Parameters:
            model (object): The model containing the weights to be regularized.
            weight_decay (float): The strength of the regularization term (default is 1).
        """
        self.model = copy.deepcopy(model)
        self.weight_decay = weight_decay

    def l0(self):
        """
        Computes the L0 norm of the model's weights.

        The L0 norm represents the number of nonzero weights, which encourages sparsity.

        Returns:
            int: The number of nonzero weights times the regularization strength.
        """
        return sum([1 for w in self.model.w if w != 0]) * self.weight_decay

    def l1(self):
        """
        Computes the L1 norm of the model's weights.

        The L1 norm is the sum of the absolute values of the weights.
        This type of regularization encourages sparsity by pushing small weights toward zero.

        Returns:
            float: The sum of the absolute values of the weights times the regularization strength.
        """
        return sum([abs(w) for w in self.model.w]) * self.weight_decay

    def l2(self):
        """
        Computes the L2 norm of the model's weights.

        The L2 norm is the sum of the squares of the weights.
        This type of regularization helps reduce overfitting by discouraging large weights.

        Returns:
            float: The sum of the squared weights times the regularization strength.
        """
        return sum([w**2 for w in self.model.w]) * self.weight_decay

    def __call__(self):
        (self.l0(), self.l1(), self.l2())

    def __str__(self):
        return f"""
        Regularization:
        L0: {self.l0()}
        L1: {self.l1()}
        L2: {self.l2()}
        """