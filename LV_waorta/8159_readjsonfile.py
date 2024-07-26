import json
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt


with open('8159_baselinebpm_hemodynamics.json', "r") as f:
    file_data = json.load(f)

# Load strain
t_strain = file_data["t_strain"]
Ecc = file_data["Ecc"]
Ell = file_data["Ell"]

# Load LV pressure
t_LVPT = file_data["t_LVPT"]
LVPT = file_data["LVPT"]

# Load AO pressure
t_AOP = file_data["t_AOPw"]
AOP = file_data["AOPw"]



