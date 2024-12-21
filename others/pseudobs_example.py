import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

import numpy as np
import pandas as pd


# Load the example planets dataset
planets = sns.load_dataset("planets")
planets.dropna(inplace=True)

# path = "C:\Users\flowb\OneDrive\TWVL\tesis_grothe\ai_cbv\new\inout"
excel_file = "nn_outputs.csv"
data = pd.read_csv("./inout/"+excel_file)
print(data)


# cond1 = planets["distance"] < 100
# planets.where( cond1, inplace=True)
# planets.where(planets["orbital_period"] < 10000 , inplace=True)

def psobs(data, x="distance", y="orbital_period", xlabel="X", ylabel="Y"):

    # data[x] = np.log(data[x])
    # data[y] = np.log(data[y])

    data[xlabel] = data[x]
    data[ylabel] = data[y]

    dist = data[x]
    period = data[y]

    psobs1 = ECDF(dist)(dist)
    psobs2 = ECDF(period)(period)
    data["U"] = psobs1
    data["V"] = psobs2


    sns.jointplot(data=data, x=xlabel, y=ylabel)
    #Show full screen
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.savefig("h0h1.png", dpi=1500, bbox_inches='tight')
    plt.show()
    
    sns.jointplot(data=data, x="U", y="V")
    #Show full screen
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.savefig("h0h1_uv.png", dpi=1500, bbox_inches='tight')
    plt.show()

# psobs(data)
# psobs(data, x="mass")


# h0 = ["h0_0", "h0_1", "h0_2", "h0_3", "h0_4", "h0_5"]
h0 = [ "h0_2",]
# h1 = ["h1_0", "h1_1", "h1_2", "h1_3", "h1_4", "h1_5"]
h1 = ["h1_0",]

for h0i in h0:
    for h1j in h1:
        x = h0i
        y = h1j
        psobs(data, x=x, y=y, xlabel=x, ylabel=y)


