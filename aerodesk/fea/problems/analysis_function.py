class Function:
    def __init__(self, name, adjoint=True):
        self.name = name
        self.adjoint = adjoint
        self.value = None
        self.derivatives = {}

    def reset(self):
        """zero the derivatives and function values"""
        self.value = 0.0
        self.derivatives = {key: 0.0 for key in self.derivatives}
        return self

    def register_to(self, problem):
        """register the function to a problem object"""
        problem.register(self)
        return self

    @classmethod
    def mass(cls):
        return cls(name="mass", adjoint=False)

    @classmethod
    def max_stress(cls):
        return cls(name="max_stress", adjoint=True)
