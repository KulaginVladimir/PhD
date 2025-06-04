from model import LID_simulation
import sys
import numpy as np

N = int(sys.argv[1])

E_dt = 1.5
eta_tr = 1e-2
phi = 1

materials = ["W", "Be"]
durations = ["ns", "us", "ms"]
E_maxs = [10.2e3, 160e3, 3.8e6]

prms = []
for mat in materials:
    for i in range(3):
        prms.append([E_maxs[i], durations[i], mat])

for i in range(1, 26):
    E_max, dur, mat = prms[N - 1]

    model = LID_simulation(
        E_max * i / 25,
        dur,
        mat,
        E_dt,
        eta_tr,
        phi,
        "../results",
        is_txt=False,
        soret=True,
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

    out = np.column_stack([Jat, Jmol_s, J_mol_sb, J_mol_b, J_des])

    filename = f"../results/fluxes_{mat}_{dur}_E{E_max/1e6:.5f}MJ_Edt{E_dt:.5f}eV_eta{-np.log10(eta_tr):.5f}.csv"
    np.savetxt(filename, out, delimiter=",")
