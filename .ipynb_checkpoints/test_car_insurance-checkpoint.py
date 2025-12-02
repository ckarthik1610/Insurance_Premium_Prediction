from Car_Insurance.training import CarInsurance

def car_test_quote():
    model = CarInsurance()
    quote = model.quote_display(
        age=25,
        annual_km=15000,
        car_age=5,
        exp_years=3,
        num_accidents=1,
        vehicle_type="suv"
    )
    print("Premium:", quote)

if __name__ == "__main__":
    car_test_quote()