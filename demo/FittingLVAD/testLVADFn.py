from LVADFn import LVAD as LVAD
HeartMate = LVAD("/mnt/home/lclee/heArt_py3/demo/FittingLVAD/HeartMate.npz")

H = -0.1000
LVADrpm = 5000
Qlvad = HeartMate.Flowrate(H, LVADrpm) / 60  # Flow rate of LVAD in mL/ms
print(Qlvad)
