from lmfit import minimize, Parameters, Parameter, report_fit
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

def dynamic_zong(y, t, theta, I):
    N0 = 10**theta[0];
    NI = 10**theta[1];
    b = theta[2];

    aT = 10**theta[3];
    bT = theta[4];
    KIT = 10**theta[5];
    nIT = theta[6];
    KlacT = 10**theta[7];

    aR = 10**theta[8];
    bR = theta[9];
    KIR = 10**theta[10];
    nIR = theta[11];
    KlacR = 10**theta[12];

    aG = 10**theta[13];
    bG = theta[14];
    KA = 10**theta[15];
    nA = theta[16];
    KR = 10**theta[17];
    nR = theta[18];

    # Gompertz growth
    Nt = N0 * np.exp(np.log(NI / N0) * (1 - np.exp(-b * t)));
    Ndt = N0 * np.exp(np.log(NI / N0) * (1 - np.exp(-b * (t+1e-3))));
    mu = (Ndt - Nt) / (1e-3 * Nt)

    dydt = [1]*3;
    dydt[0] = aT * mu * (1 + (I / KIT) ** nIT) / (1 + (I / KIT) ** nIT + KlacT) + bT * mu - mu * y[0]; # T7
    dydt[1] = aR * mu * (1 + (I / KIR) ** nIR) / (1 + (I / KIR) ** nIR + KlacR) + bR * mu - mu * y[1]; # R
    dydt[2] = aG * mu * (y[0] ** nA / (KA ** nA + y[0] ** nA)) * (KR ** nR / (KR ** nR + y[1] ** nR)) + bG * mu - mu * y[2]; # GFP

    return dydt


def g(t, y0, theta, I):
    x = odeint(dynamic_zong, y0, t, args=(theta, I, ))
    return x


def residual(theta, ts, data):
    y0 = theta['T0'].value, theta['R0'].value, theta['GFP0'].value
    model = g(ts, y0, theta, I)
    return (model - data).ravel()

t = np.linspace(0, 20, 100)
I = 30
y0 = np.array([1, 1, 1])

N0 = 1E7
NI = 1E9
b = 0.2

aT = 6223
bT = 12.8
KIT = 1.4*10**-6
nIT = 2.3
KlacT = 15719

aR = 8025
bR = 30.6
KIR = 1.2*10**-6
nIR = 2.2
KlacR = 14088

aG = 16462
bG = 19
KA = 2532
nA = 1.34
KR = 987
nR = 3.9

true_params = np.array((np.log10(N0), np.log10(NI), b, np.log10(aT), bT, np.log10(KIT), nIT, np.log10(KlacT), np.log10(aR),
                        bR, np.log10(KIR), nIR, np.log10(KlacR), np.log10(aG), bG, np.log10(KA), nA, np.log10(KR), nR))

data = g(t, y0, true_params, I)

# set parameters incluing bounds
params = Parameters()
params.add('T0', value=float(data[0, 0]), min=0, max=10)
params.add('R0', value=float(data[0, 1]), min=0, max=10)
params.add('GFP0', value=float(data[0, 1]), min=0, max=10)
params.add('N0', value=6.0, min=4.0, max=8.0)
params.add('NI', value=9.0, min=8.0, max=10.0)
params.add('b', value=0.1, min=0, max=1)
params.add('aT', value=4.0, min=2.0, max=5.0)
params.add('bT', value=10, min=0, max=100)
params.add('KIT', value=-6.0, min=-8.0, max=-5.0)
params.add('nIT', value=1, min=0, max=3)
params.add('KlacT', value=4.0, min=3.0, max=5.0)
params.add('aR', value=4.0, min=2.0, max=5.0)
params.add('bR', value=10, min=0, max=100)
params.add('KIR', value=-6.0, min=-8.0, max=-5.0)
params.add('nIR', value=1, min=0, max=3)
params.add('KlacR', value=4.0, min=3.0, max=5.0)
params.add('aG', value=4.0, min=2.0, max=5.0)
params.add('bG', value=10, min=0, max=100)
params.add('KA', value=3.0, min=2.0, max=5.0)
params.add('nA', value=1, min=0, max=4)
params.add('KR', value=3.0, min=2.0, max=5.0)
params.add('nR', value=1, min=0, max=4)

# fit model and find predicted values
result = minimize(residual, params, args=(t, data), method='leastsq')
final = data + result.residual.reshape(data.shape)

# plot data and fitted curves
plt.plot(t, data[:,2], 'o')
plt.show()
# plt.plot(t, final, '-', linewidth=2);

# display fitted statistics
# report_fit(result)
