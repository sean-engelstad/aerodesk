__all__ = ["Isotropic"]


class Isotropic:
    def __init__(self, E=1e7, rho=2.7e3):
        self.E = E
        self.rho = rho

    @classmethod
    def aluminum_6061(cls):
        """aluminum 6061 alloy"""
        return cls(E=70e9, rho=2.7e3)

    @classmethod
    def test_material(cls):
        """material with lower E,rho for unittests"""
        return cls(E=1e7, rho=0.1)
