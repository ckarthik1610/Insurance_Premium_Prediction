import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Car_Insurance.training import result

def test_result_function():
    print("Running result() test...")
    
    premium = result(
        age=18,
        annual_km=20000,
        car_age=12,
        exp_years=2,
        num_accidents=0,
        vehicle_type="suv"
    )
    
    print("===========================================")
    print("        CAR INSURANCE RESULT() TEST")
    print("===========================================")
    print(f"Premium Returned from result():   {premium}")
    print("===========================================")


if __name__ == "__main__":
    test_result_function()
#from Car_Insurance.training import result
#
#print(result(18, 20000, 12, 2, 0, "suv"))