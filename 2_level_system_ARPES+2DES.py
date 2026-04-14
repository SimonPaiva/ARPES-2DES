# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 09:59:28 2026

@author: simon
"""

from qutip import *
import numpy as np 
from qudpy.Classes import *
import matplotlib.pyplot as plt
import qudpy.plot_functions as pf
from qutip import Qobj
import ufss  # diagram generation
wa1=1
wb1=1.03
wc1=1.06
wd1=1.09


a = destroy(2)
b = destroy(2)
c = destroy (2)
d = destroy (2)
I = qeye(2)
a =  tensor(I,tensor(I,tensor(I,a)))
b =  tensor(I,tensor(I,tensor(b,I)))
c =  tensor(I,tensor(c,tensor(I,I)))
d =  tensor(d,tensor(I,tensor(I,I)))



Hs =  wa1*a.dag()*a+wb1*b.dag()*b+wc1*c.dag()*c+wd1*d.dag()*d



# H_i = 0
H = Hs

# %%


coher_l = np.sqrt(0.001)
relax_rate = np.sqrt(0.005)
mix = np.sqrt(0.01)
L_d_a = a.dag()*a*coher_l
L_d_b = b.dag()*b*coher_l
L_d_c = c.dag()*c*coher_l
L_d_d = d.dag()*d*coher_l


L_r_a = a*relax_rate
L_r_b = b*relax_rate
L_r_c = c*relax_rate
L_r_d = d*relax_rate


L_mix_dc = mix*(d.dag()*c+c.dag()*d)
L_mix_db = mix*(d.dag()*b+b.dag()*d)
L_mix_da = mix*(d.dag()*a+a.dag()*d)
L_mix_cb = mix*(c.dag()*b+b.dag()*c)
L_mix_ca = mix*(c.dag()*a+a.dag()*c)
L_mix_ba = mix*(b.dag()*a+a.dag()*b)

# %%

c_ops = [L_d_a,L_d_b,L_d_c,L_d_d,L_r_a,L_r_b,L_r_c,L_r_d,L_mix_dc,L_mix_db,L_mix_da,L_mix_cb,L_mix_ca,L_mix_ba]
en,T = H.eigenstates()

Hd=H.transform(T)

A = a.transform(T)


B = b.transform(T)

C = c.transform(T)

D = d.transform(T)

AB = A+B+C+D

# AB = mu_t.transform(T)
# AB = mu_t


# c_ops_d = c_ops
c_ops_d = [c.transform(T) for c in c_ops]
# mud_A = A1.dag()* A1 
# mud_B = B1.dag()*B1 
    
# mud_A = A2.dag()* A2
# mud_B = B2.dag()*B2

N_exc = (
    A.dag()*A 
)

# N_exc=a1.dag()*a1+a2.dag()*a2+b1.dag()*b1+b2.dag()*b2

mud =N_exc


rhoSS=steadystate(Hd, c_ops_d)
# rhoSS = np.zeros((6,6))
# rhoSS[0,0]=1
# rhoSS =Qobj(rhoSS)
# %%

DG = ufss.DiagramGenerator
Q4th = DG(detection_type='fluorescence')  # to determine the diagrams for action spectroscopies, use detection_type = 'fluorescence'

# Set the pulse duration
t_pulse = np.array([-1, 1])  # pulse duration 2fs
Q4th.efield_times = [t_pulse]*4  # same pulse duration for all 4 pulses

Q4th.set_phase_discrimination([(0, 1), (1, 0), (1, 0), (0, 1)])  # rephasing

[Q8b, Q2b, Q3a, Q4a] = Q4th.get_diagrams([0,100,200,300])
rephasing = [Q3a, Q4a, Q2b, Q8b]
print('the rephasing diagrams are -Q3a (SE2), -Q4a (GSB2), -Q2b* (ESA2), and Q8b* (λ.ESA2) ', rephasing)


Q4th.set_phase_discrimination([(1, 0), (0, 1), (1, 0), (0, 1)])  # non-rephasing

[Q7b, Q3b, Q2a, Q5a] = Q4th.get_diagrams([0,100,200,300])
nonrephasing = [Q2a, Q5a, Q3b, Q7b]
print('the nonrephasing diagrams are -Q2a (SE1), -Q5a* (GSB1), -Q3b* (ESA1), and Q7b (λ.ESA1) ', nonrephasing)

Q4th.set_phase_discrimination([(0, 1), (1, 0), (1, 0), (0, 1)])  # double quantum coherence
Q4th.maximum_manifold = 1
[M1,M2] = Q4th.get_diagrams([0,100,200,300])
re_nonre_m1 = [M1,M2]


Q4th.set_phase_discrimination([(1, 0), (0, 1), (1, 0), (0, 1)])  # double quantum coherence
Q4th.maximum_manifold = 1
[M3,M4] = Q4th.get_diagrams([0,100,200,300])
re_nonre_m1.append(M3)
re_nonre_m1.append(M4)


Reso =  1.5/np.pi

###System Variables####
sys2 = System(H=Hd, a=AB, u=mud, c_ops=c_ops_d, rho=rhoSS, diagonalize=False)
sys2.hbar=1


time_delays = [351,40,351,0]
scan_id = [0, 2]
response_list = []
states_list = []
diagrams = rephasing+nonrephasing


for k in range(len(diagrams)):
    states, t1, t2, dipole = sys2.coherence2d(time_delays, diagrams[k], scan_id, r=Reso)
    print('diagram ', k, ' done')
    response_list.append(dipole)
    states_list.append(states)


spectra_list, extent, f1, f2 = sys2.spectra(np.imag(response_list),resolution=Reso)


delta = 1
rephasing_spectra = spectra_list[:4]

sum_=np.sum(spectra_list[:2],0)-delta*spectra_list[2]+spectra_list[3]
rephasing_spectra.append(sum_)
# pf.silva_plot(rephasing_spectra, f1,f2, labels=['E emission', 'E absorption'],
#               scale='linear', color_map='bwr',
#               interpolation='spline36', center_scale=False, plot_sum=False, plot_quadrant='2', invert_y=False,
#               diagonals=[True, False])

pf.silva_plot_contourf(rephasing_spectra, f1,f2, labels=['E emission', 'E absorption'],
              scale='linear', color_map='jet',
              center_scale=False, plot_sum=False, plot_quadrant='Zoom', invert_y=False,
              diagonals=[True, False],nlevels=30,Zoom_coor=[0.9,1.3,-1.3,-0.9])

nonrephasing_spectra = spectra_list[4:]
sum_ = np.sum(spectra_list[5:],0)
sum_ += -delta*spectra_list[4]
nonrephasing_spectra.append(sum_)
# pf.silva_plot(nonrephasing_spectra, f1,f2, labels=['E emission', 'E absorption'],
#               scale='linear', color_map='bwr',
#               interpolation='spline36', center_scale=False, plot_sum=False, plot_quadrant='1', invert_y=False,
#               diagonals=[False, True])

pf.silva_plot_contourf(nonrephasing_spectra, f1,f2, labels=['E emission', 'E absorption'],
              scale='linear', color_map='jet',
              center_scale=False, plot_sum=False, plot_quadrant='Zoom', invert_y=False,
              diagonals=[False, True],nlevels=30,Zoom_coor=[0.9,1.3,0.9,1.3])

response_list_ = []
states_list_ = []
for k in range(len(re_nonre_m1)):
    states, t1, t2, dipole = sys2.coherence2d(time_delays, re_nonre_m1[k], scan_id, r=Reso)
    print('diagram ', k, ' done')
    response_list_.append(dipole)
    states_list_.append(states)


spectra_list, extent, f1, f2 = sys2.spectra(np.imag(response_list_),resolution=Reso)
rephasing_spectra = spectra_list[:2]

sum_=np.sum(rephasing_spectra,0)
rephasing_spectra.append(sum_)
# pf.silva_plot(rephasing_spectra, f1,f2, labels=['E emission', 'E absorption'],
#               scale='linear', color_map='bwr',
#               interpolation='spline36', center_scale=False, plot_sum=False, plot_quadrant='2', invert_y=False,
#               diagonals=[True, False])

pf.silva_plot_contourf(rephasing_spectra, f1,f2, labels=['E emission', 'E absorption'],
              scale='linear', color_map='jet',
              center_scale=False, plot_sum=False, plot_quadrant='Zoom', invert_y=False,
              diagonals=[True, False],nlevels=30,Zoom_coor=[0.9,1.3,-1.3,-0.9])

nonrephasing_spectra = spectra_list[2:]
sum_ = np.sum(nonrephasing_spectra, 0)
nonrephasing_spectra.append(sum_)
# pf.silva_plot(nonrephasing_spectra, f1,f2, labels=['E emission', 'E absorption'],
#               scale='linear', color_map='bwr',
#               interpolation='spline36', center_scale=False, plot_sum=False, plot_quadrant='1', invert_y=False,
#               diagonals=[False, True])

pf.silva_plot_contourf(nonrephasing_spectra, f1,f2, labels=['E emission', 'E absorption'],
              scale='linear', color_map='jet',
              center_scale=False, plot_sum=False, plot_quadrant='Zoom', invert_y=False,
              diagonals=[False, True],nlevels=30,Zoom_coor=[0.9,1.3,0.9,1.3])

