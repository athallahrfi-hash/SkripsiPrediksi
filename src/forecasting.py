import pandas as pd


def moving_average_forecast(data, n=3):
    forecast = data["Jumlah Standar"].tail(n).mean()

    return round(forecast, 2)
