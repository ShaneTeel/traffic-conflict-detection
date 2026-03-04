import numpy as np

def root_mean_squared_error(Geo_true, Geo_pred):
    col_width = 20
    error_str = f"\n{"Point #":<{col_width}}{"True East":<{col_width}}{"Pred. East":<{col_width}}{"East Error^2":<{col_width}}{"True North":<{col_width}}{"Pred. North":<{col_width}}{"North Error^2":<{col_width}}{"Total Error":<{col_width}}\n"
    popup_dict = {}
    sum_squared_error = []
    i = 0
    
    for (x1, y1), (x2, y2) in zip(Geo_true, Geo_pred):
        x_dist = (x2 - x1)**2
        y_dist = (y2 - y1)**2
        sum_squared_dist = x_dist + y_dist
        sum_squared_error.append(sum_squared_dist)

        popup_dict[i] = { 
            "Pred. East": x2,
            "East Error^2": x_dist,
            "Pred. North": y2,
            "North Error^2": y_dist,
            "Total Error": sum_squared_dist
        }

        error_str += f"{i:<{col_width}}{x1:<{col_width}}{x2:<{col_width}}{x_dist:<{col_width}.2f}{y1:<{col_width}}{y2:<{col_width}}{y_dist:<{col_width}.2f}{sum_squared_dist:<{col_width}.2f}\n"
        i += 1

    rmse = np.sqrt(np.mean(sum_squared_error))
    error_str = f"\nRoot Mean Squared Error: {rmse:.2f}\n" + error_str
    return error_str, popup_dict

def mean_absolute_average(Geo_true, Geo_pred):
    col_width = 20
    error_str = f"\n{"Point #":<{col_width}}{"True East":<{col_width}}{"Pred. East":<{col_width}}{"East |Error|":<{col_width}}{"True North":<{col_width}}{"Pred. North":<{col_width}}{"North |Error|":<{col_width}}{"Total Error":<{col_width}}\n"
    popup_dict = {}
    sum_error = []
    i = 0

    for (x1, y1), (x2, y2) in zip(Geo_true, Geo_pred):
        x_dist = abs(x2 - x1)
        y_dist = abs(y2 - y1)
        sum_dist = x_dist + y_dist
        sum_error.append(sum_dist)

        popup_dict[i] = {
            "Pred. East": x2,
            "East |Error|": x_dist,
            "Pred. North": y2,
            "North |Error|": y_dist,
            "Total Error": sum_dist
        }

        error_str += f"{i:<{col_width}}{x1:<{col_width}}{x2:<{col_width}}{x_dist:<{col_width}.2f}{y1:<{col_width}}{y2:<{col_width}}{y_dist:<{col_width}.2f}{sum_dist:<{col_width}.2f}\n"
        i += 1

    mae = np.mean(sum_error)
    error_str = f"\nMean Absolute Error: {mae:.2f}\n" + error_str
    return error_str, popup_dict