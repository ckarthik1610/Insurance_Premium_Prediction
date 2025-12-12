"""
training.py
------------------------------------
This module calculates the final car insurance premium by combining all the risk factors present in preprocessing.py. Here we can notice inheritance -> CarInsurance extends the parent class Insurance.
"""
from . import preprocessing
from .exceptions import InvalidInputError

class Insurance:
    """ Parent class storing the base_cost. """
    def __init__(self, base_cost=493.74225):
        if base_cost <= 0:
            raise InvalidInputError("Base cost must be positive.")
        self.base_cost = base_cost

    def yearly_premium(self):
        """Returns the base premium before risk adjustments."""
        return self.base_cost

class CarInsurance(Insurance):
    """ Child class extending Insurance. """
    Vehicle_Type_Multipliers = {
        "sedan": 1.00,
        "suv": 1.05,
        "sports": 1.15,
        "truck": 1.08
    }

    def __init__(self, base_cost=493.74225):
        super().__init__(base_cost)

    def vehicle_type_factor(self, vehicle_type):
        """Return vehicle type multiplier."""
        if vehicle_type is None:
            return 1.0
        if not isinstance(vehicle_type, str):
            raise InvalidInputError("Vehicle type must be a string.")
            
        return self.Vehicle_Type_Multipliers.get(vehicle_type.lower(), 1.0)
    
    def total_risk(self,age,annual_km,car_age,year_driving,num_accidents,vehicle_type=None,):
        """ Combining all risk factors. """
        try:
            base_risk = preprocessing.combined_factors(age, annual_km, car_age, year_driving, num_accidents)
            type_risk = self.vehicle_type_factor(vehicle_type)
        except InvalidInputError as e:
            raise InvalidInputError(f"Cannot compute risk: {e}")

        return type_risk * base_risk
    
    def final_premium(self,age,annual_km,car_age,exp_years,num_accidents,vehicle_type=None,):
        """ base premium × combined risk """
        risk = self.total_risk(age, annual_km, car_age, exp_years, num_accidents, vehicle_type)
        premium = round(self.base_cost * risk, 2)

        if premium < 0:
            raise InvalidInputError("Final premium cannot be negative.")
        return premium
    
    def quote_display(self,age,annual_km,car_age,exp_years,num_accidents,vehicle_type=None,):
        """ Summary """
        return self.final_premium(age, annual_km, car_age, exp_years, num_accidents, vehicle_type)
    
def result(age, annual_km, car_age, exp_years, num_accidents, vehicle_type):
    """
    Convenience helper to quickly compute premium.
    """
    model = CarInsurance()
    return model.quote_display(
        age=age,
        annual_km=annual_km,
        car_age=car_age,
        exp_years=exp_years,
        num_accidents=num_accidents,
        vehicle_type=vehicle_type,
    )
    
    
