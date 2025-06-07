from model import LID_simulation
import sys
import numpy as np

N = int(sys.argv[1])

E_dts = [1.0, 1.25, 1.5, 1.75, 2.0]
E_dt = 1.5

eta_trs = [1e-4, 1e-3, 1e-2, 1e-1]
eta_tr = 1e-2

phis = [1, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3]
phi = 1

materials = ["W", "Be"]
durations = ["ns", "us", "ms"]
E_maxs = [10.2e3, 160e3, 3.8e6]

soret = [True, False]
prms = []

for sor in soret:
    for i in range(3):
        prms.append([E_maxs[i], durations[i], E_dt, eta_tr, phi, sor])

mat = "W"
output_folder = "../results",
for i in range(1, 26):
    print("\n")
    print(f"Iteration: {i}")
    
    E_max, dur, E_dt, eta_tr, phi, sor = prms[N-1] 


    E = E_max * i / 25
    model = LID_simulation(
        E,
        dur,
        mat,
        E_dt,
        eta_tr,
        phi,
        "../results_soret/",
        is_txt=True,
        soret=sor,
    )

    results = model.run()

    t = np.array(results.t)
    cs = np.array(results[0].data)
    T = np.array(results[2].data)
    cm = np.array(results[3].data)

    Jat = model.J_a(T, cs)
    Jmol_s = model.J_m_s(T, cs)
    J_mol_sb = 2 * model.J_m_sb(T, cs, cm)
    J_mol_b = model.J_m_b(T, cm)

    J_des = Jat + Jmol_s + J_mol_b + J_mol_sb

    out = np.column_stack([t, Jat, Jmol_s, J_mol_sb, J_mol_b, J_des])

    if sor:
        filename = f"../results_soret/fluxes_{mat}_{dur}_E{E/1e6:.5f}MJ_Edt{E_dt:.5f}eV_eta{-np.log10(eta_tr):.5f}_phi{-np.log10(phi):.5f}_wSoret.csv"
    else:
        filename = f"../results_soret/fluxes_{mat}_{dur}_E{E/1e6:.5f}MJ_Edt{E_dt:.5f}eV_eta{-np.log10(eta_tr):.5f}_phi{-np.log10(phi):.5f}_woSoret.csv"
        
    np.savetxt(filename, out, delimiter=",")