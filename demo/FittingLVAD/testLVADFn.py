from LVADFn import LVAD as LVAD
HeartMate = LVAD("HeartMate.npz")

H = 43#-0.1000
LVADrpm = 5000#5000
Qlvad = HeartMate.Flowrate(H, LVADrpm) / 60  # Flow rate of LVAD in mL/ms
print(Qlvad)
