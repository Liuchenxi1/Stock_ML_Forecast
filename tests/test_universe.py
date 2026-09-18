from stock_ml_forecast.universe import get_sp500_universe


def main():
    universe = get_sp500_universe()

    print(universe.head())
    print(len(universe))


if __name__ == "__main__":
    main()