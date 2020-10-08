import load_data

if __name__ == "__main__":
    load_data.fetch_housing_data()
    print("in another.py")
    print(load_data.HOUSING_PATH)