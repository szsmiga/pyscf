#!/usr/bin/env python
"""Compute relaxed CCSD(T) dipole for CO/cc-pVTZ.

Reference setup:
- CO bond length: R(CO) = 1.128 Angstrom (O at z = -1.128)
- Basis: cc-pVTZ
- Property: dipole moment from relaxed CCSD(T) 1-RDM

Expected |mu_z| is approximately 0.1703 Debye.
"""

from pyscf import gto, scf
from pyscf.cc import ccsd_t_lambda, ccsd_t_rdm


def main():
    mol = gto.M(
        atom='C 0.0 0.0 0.0; O 0.0 0.0 -1.128',
        basis='cc-pvtz',
        unit='Angstrom',
        verbose=4,
    )

    mf = scf.RHF(mol).run(conv_tol=1e-12)
    mycc = mf.CCSD().run(conv_tol=1e-10, conv_tol_normt=1e-8)

    t1, t2 = mycc.t1, mycc.t2
    eris = mycc.ao2mo()
    _, l1, l2 = ccsd_t_lambda.kernel(mycc, eris, t1, t2)

    dm1_relaxed_ao = ccsd_t_rdm.make_rdm1(
        mycc, t1, t2, l1, l2, eris=eris, ao_repr=True, relaxed=True
    )

    dip = mf.dip_moment(mol, dm1_relaxed_ao, unit='Debye', verbose=4)

    print('\nRelaxed CCSD(T) dipole (Debye):', dip)
    print('abs(mu_z) = %.6f Debye' % abs(dip[2]))


if __name__ == '__main__':
    main()
