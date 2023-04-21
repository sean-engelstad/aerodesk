__all__ = ["ThicknessVariable"]


class ThicknessVariable:
    def __init__(
        self,
        name,
        base,
        thickness,
        inertia_shape_factor=1.0,
        area_factor=1.0,
        active=False,
    ):
        self.name = name
        self.base = base
        self.thickness = thickness
        self.inertia_shape_factor = inertia_shape_factor
        self.area_factor = area_factor
        self.active = active

    @property
    def value(self) -> float:
        return self.thickness

    @value.setter
    def value(self, new_value):
        self.thickness = new_value
        return

    @classmethod
    def from_inertia(cls, I):
        """since we are not optimizing this guy we will"""
        base = 1
        thickness = (12 * I / base) ** (1.0 / 3)
        return cls(base=base, thickness=thickness, active=False)

    @classmethod
    def calculate_inertia(cls, base, height):
        return base * height**3 / 12.0

    @classmethod
    def dinertia_dh(cls, base, height):
        return base * height**2 / 4.0

    @property
    def bending_inertia(self):
        return self.inertia_shape_factor * ThicknessVariable.calculate_inertia(
            base=self.base, height=self.thickness
        )

    @property
    def dIdh(self):
        return self.inertia_shape_factor * ThicknessVariable.dinertia_dh(
            base=self.base, height=self.thickness
        )

    @property
    def area(self):
        return self.base * self.thickness * self.area_factor

    @property
    def dAdh(self):
        return self.base * self.area_factor

    @classmethod
    def test_inertia_jacobian(cls, base, height):
        deriv = cls.dinertia_dh(base, height)
        import numpy as np

        h = 1e-30
        dIds = np.imag(cls.calculate_inertia(base, height + 1j * h)) / h
        rel_error = (deriv - dIds) / dIds
        return rel_error
