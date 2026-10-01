import matplotlib.pyplot as plt
import astropy.units as u
from trm import roche

# Input data
K2 = 393.75 # donor RV amplitude
K1 = 68.75 # WD RV amplitude
q = K1 / K2 # mass-ratio
a = 0.549*u.solRad # orbital separation
Rwd = 0.0133*u.solRad # WD radius

# Roche lobe

rx1, ry1 = roche.lobe1(q)
rx2, ry2 = roche.lobe2(q)

# Roche lobe (velocity space)

vx,vy = roche.vlobe2(q)
vx *= K1+K2
vy *= K1+K2

vx_wd,vy_wd = roche.vlobe1(q)
vx_wd *= K1+K2
vy_wd *= K1+K2

# Accretion stream

vx1_st, vy1_st = roche.vstream(q, step=0.01, vtype=1, n=75)
vx1_st *= K1+K2
vy1_st *= K1+K2

vx2_st, vy2_st = roche.vstream(q, step=0.01, vtype=2, n=85)
vx2_st *= K1+K2
vy2_st *= K1+K2

fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12,6))


ax[0].set_title("Roche radii")
ax[0].plot(rx1*a, ry1*a)
ax[0].plot(rx2*a, ry2*a)
ax[0].set_xlabel("X [solar radii]")
ax[0].set_ylabel("Y [solar radii]")

ax[1].set_title("Velocity space")
ax[1].plot(vx, vy, 'r')
ax[1].plot(vx_wd, vy_wd, 'b')
ax[1].plot(vx1_st, vy1_st, 'r')
ax[1].plot(vx2_st, vy2_st, 'r')
ax[1].set_xlabel("Vx [km/s]")
ax[1].set_ylabel("Vy [km/s]")

plt.savefig("roche_velocities.png", dpi=300)