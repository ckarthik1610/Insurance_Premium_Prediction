"""
preprocessing.py
----------------------------------------
This model is used to calculate the individual risk factors used in the car insurance pricing system. The multipliers are used to estimate overall customer risk, which affects the final premium.
"""
from .exceptions import InvalidInputError
# Real Age Multipliers
Age_Multipliers = {
    18: 1.018341, 19: 1.016126, 20: 1.016307, 21: 1.017236, 22: 1.014096,
    23: 1.013065, 24: 1.010331, 25: 1.011745, 26: 1.011673, 27: 1.009503,
    28: 1.010302, 29: 1.006072, 30: 1.007313, 31: 1.007149, 32: 1.005220,
    33: 1.004939, 34: 1.003887, 35: 1.005085, 36: 1.001125, 37: 1.005335,
    38: 1.002755, 39: 1.001005, 40: 1.000958, 41: 1.000113, 42: 1.000244,
    43: 0.995342, 44: 0.996376, 45: 0.994429, 46: 0.995000, 47: 0.995034,
    48: 0.996035, 49: 0.994798, 50: 0.994178, 51: 0.994671, 52: 0.993242,
    53: 0.991606, 54: 0.991392, 55: 0.992380, 56: 0.987896, 57: 0.990866,
    58: 0.991267, 59: 0.988257, 60: 0.989355, 61: 0.987712, 62: 0.984591,
    63: 0.986329, 64: 0.988128, 65: 0.988006
}
def age_factor(age):
    if age is None or not isinstance(age,(int,float)):
        raise InvalidInputError("Age must be numeric.")
    if age < 0:
        raise InvalidInputError("Age cannot be negative.")
    return Age_Multipliers.get(age, 1.0)

# Real Mileage Multiplers
Annual_Mileage_Multipliers = {
    11: 1.001179, 12: 0.998162, 13: 0.999003, 14: 0.998490, 15: 1.001794,
    16: 0.999642, 17: 1.000345, 18: 1.000652, 19: 1.000291, 20: 0.999599,
    21: 0.999225, 22: 0.998437, 23: 1.000596, 24: 1.000745, 25: 1.001651
}
def mileage_factor(annual_km):
    if annual_km is None or not isinstance(annual_km, (int, float)):
        raise InvalidInputError("Mileage must be numeric.")
    if annual_km < 0:
        raise InvalidInputError("Mileage cannot be negative.")
    key = int(round(annual_km / 1000))
    return Annual_Mileage_Multipliers.get(key, 1.0)

# Car Multiplier multiplier
Car_Age_Multiplier = {
    0: 0.996161,  1: 0.997162,  2: 0.996400,  3: 0.995873,
     4: 0.999774,  5: 0.998365,  6: 0.998602,  7: 0.998457,
     8: 0.996424,  9: 0.996404, 10: 0.997755, 11: 1.000117,
    12: 1.002003, 13: 0.996691, 14: 0.996941, 15: 1.002600,
    16: 0.999387, 17: 1.003647, 18: 1.000895, 19: 0.999384,
    20: 1.001377, 21: 0.999499, 22: 1.001073, 23: 0.999233,
    24: 0.999879, 25: 1.000636, 26: 1.003596, 27: 1.001766,
    28: 1.001361, 29: 0.999855, 30: 1.001280, 31: 1.001400,
    32: 1.003001, 33: 1.003452, 34: 1.006315, 35: 1.005208
}
def car_age_factor(car_age):
    if car_age is None or not isinstance(car_age, (int, float)):
        raise InvalidInputError("Car age must be numeric.")
    if car_age < 0:
        raise InvalidInputError("Car age cannot be negative.")
    return Car_Age_Multiplier.get(car_age, 1.0)

# Experience Multipler multiplier
Experience_Multiplier = {
     0: 1.014075,  1: 1.011386,  2: 1.012660,  3: 1.012247,  4: 1.009931,
     5: 1.008174,  6: 1.008124,  7: 1.006879,  8: 1.004306,  9: 1.004697,
    10: 1.002998, 11: 1.003466, 12: 1.001825, 13: 1.002492, 14: 1.000973,
    15: 1.001573, 16: 0.999480, 17: 1.000299, 18: 0.997058, 19: 0.993784,
    20: 0.995642, 21: 0.993829, 22: 0.995090, 23: 0.993205, 24: 0.991134,
    25: 0.990990, 26: 0.988596, 27: 0.989020, 28: 0.985880, 29: 0.986262,
    30: 0.987357, 31: 0.983914, 32: 0.985456, 33: 0.987762, 34: 0.982718,
    35: 0.978301, 36: 0.980407, 37: 0.979585, 38: 0.979205, 39: 0.973556,
    40: 0.979256
}
def experience_factor(year_driving):
    if year_driving is None or not isinstance(year_driving, (int, float)):
        raise InvalidInputError("Experience must be numeric.")
    if year_driving < 0:
        raise InvalidInputError("Experience cannot be negative.")
    return Experience_Multiplier.get(year_driving, 1.0)

# Accident Multiplier
Accident_Multiplier = {
    0: 0.992388, 1: 0.996378,
    2: 0.997017, 3: 1.001607,
    4: 1.003906, 5: 1.007272
}
def accident_factor(num_accidents):
    if num_accidents is None or not isinstance(num_accidents, (int, float)):
        raise InvalidInputError("Accidents must be numeric.")
    if num_accidents < 0:
        raise InvalidInputError("Accident count cannot be negative.")
    return Accident_Multiplier.get(num_accidents, 1.0)

# Total combined risks
def combined_factors(age, annual_km, car_age, year_driving, num_accidents):
    """
    Combining all individual risk multipliers into a single risk score. This will be used inside CarInsurance class.
    """
    try:
        a = age_factor(age)
        mil = mileage_factor(annual_km)
        c = car_age_factor(car_age)
        exp = experience_factor(year_driving)
        aci = accident_factor(num_accidents)
        return a * mil * c * exp * aci
    except InvalidInputError as e:
        raise InvalidInputError(f"Invalid data passed to combined_factors: {e}")
    