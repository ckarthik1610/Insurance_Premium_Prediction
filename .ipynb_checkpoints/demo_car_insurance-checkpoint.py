from car_insurance.training import CarInsurance
car_model = CarInsurance()
age = 35
annual_km = 15000
car_age = 5
exp_years = 10
num_accidents = 1
vehicle_type = "suv"
premium = car_model.final_premium(
    age,
    annual_km,
    car_age,
    exp_years,
    num_accidents,
    vehicle_type
)
print("\nFULL QUOTE OUTPUT:")
print(car_model.quote_display(
    age,
    annual_km,
    car_age,
    exp_years,
    num_accidents,
    vehicle_type
))