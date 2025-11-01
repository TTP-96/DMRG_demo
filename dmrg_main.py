#!/usr/bin/env python3
"""
Minimal two-site DMRG for the 1D Transverse-Field Ising Model (OBC)
with comparison to exact diagonalization (ED) for small N.

Pedagogical implementation - not optimized.
It shows the essential steps in DMRG: building an MPO, sweeping with two-site
updates (solve effective eigenproblem, SVD + truncate), and evaluating
observables and energy. Comparison is then made to ED results.

"""
from __future__ import annotations
import numpy as np
from numpy.linalg import svd
from dataclasses import dataclass
from typing import List, Tuple
from scipy.sparse import csr_matrix, kron as spkron, identity as spidentity
from scipy.sparse.linalg import eigsh, LinearOperator

import argparse

np.set_printoptions(precision=6, suppress=True)


# Set up Paulis

def spin_ops():
    I = np.eye(2)
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    Z = np.array([[1.0, 0.0], [0.0, -1.0]])
    return I, X, Z


# TFIM Hamiltonian as MPO (bond-dim 3)
# H = -J sum Z_i Z_{i+1} - h sum X_i
# OBC

def tfim_mpo(N: int, J: float, h: float) -> List[np.ndarray]:
    """
    OBC MPO for TFIM: H = -J Σ Z_i Z_{i+1} - h Σ X_i

    H = W[1] W[2] ... W[N]

      W[1] = [ -hX, Z, I ]            shape is  (1×3)

    Bulk tensor W is (3x3)

    W[i] = [[ I, 0, 0 ],               
               [ -JZ, 0, 0 ],
               [ -hX, Z, I ]]

     W[N] = [ I, -JZ, -hX ]^T           shape is (3×1)

    Can abbreviate as
    
    W[1] = [ C, A, I ] 
    W[i] = [[ I, 0, 0 ], 
    [ B, 0, 0 ],
    [ C, A, I ]]
    
    W[N] = [ I, B, C ]^T 
    
    with A=Z, B=-J Z, C=-h X.

    You can think of the MPO construction process as building up
    a row vector which shall be multiplied by the right boundary column vector to yield the Hamiltonian as
    a sum of products of local operators. An alternative nice way of intuiting this construction is to think in terms of finite-state automata,
    where the internal MPO bond indices correspond to states, and the local operator tensors correspond to transitions between states with associated operators.
    The states can be interpreted as keeping track of what terms in the Hamiltonian have been started but not yet completed as we move along the chain.
    At each product factor we consider what the memory of the previous factors is (the left bond index), and what new terms can be started or completed (the right bond index), with the local operators acting as the "actions" that facilitate these transitions.
    
    """
    I, X, Z = spin_ops()
    d = 2
    Aop = Z
    Bop = -J * Z
    Cop = -h * X

    W: List[np.ndarray] = []
    # Left boundary (1×3)
    W0 = np.zeros((1, 3, d, d))
    W0[0, 0] = Cop  # C
    W0[0, 1] = Aop  # A
    W0[0, 2] = I    # I
    W.append(W0)

    # Bulk (3×3)
    Wbulk = np.zeros((3, 3, d, d))
    Wbulk[0, 0] = I       # I
    Wbulk[1, 0] = Bop     # B
    Wbulk[2, 0] = Cop     # C
    Wbulk[2, 1] = Aop     # A
    Wbulk[2, 2] = I       # I
    for _ in range(N - 2):
        W.append(Wbulk.copy())

    # Right boundary (3×1)
    WN = np.zeros((3, 1, d, d))
    WN[0, 0] = I     # I
    WN[1, 0] = Bop   # B
    WN[2, 0] = Cop   # C
    W.append(WN)

    return W


# Random MPS initialization with target bond dimension m
# A[n] has shape (chiL, d, chiR) with chiL[0]=chiR[N-1]=1
# Left-canonicalize with QR and cap bonds at m


def random_mps(N: int, d: int, m: int, seed: int = 0) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    # Start with moderate internal bond dims (<= m)
    chis = [1]
    for n in range(1, N):
        chi_n=min(m, 2 ** min(n, N - n)) # grow then shrink
        if (verbose):
            print("Bond", n, "dim", chi_n)
        chis.append(chi_n)

        
    chis[-1] = 1
    A = []
    for n in range(N):
        # generate random mps
        A.append(rng.normal(size=(chis[n], d, chis[n + 1] if n < N - 1 else 1)))
        
    # Left-canonicalize and cap to m
    for n in range(N - 1):
        chiL, dloc, chiR = A[n].shape
        M = A[n].reshape(chiL * dloc, chiR) # merge left dims
        Q, R = np.linalg.qr(M) # qr decomp
        r = min(Q.shape[1], m) # cap right
        Q = Q[:, :r]           # bond dim to m
        R = R[:r, :]
        A[n] = Q.reshape(chiL, dloc, r)
        # absorb R into next tensor (left-multiply on its left bond)
        chiL2, d2, chiR2 = A[n + 1].shape
        A[n + 1] = (R @ A[n + 1].reshape(chiL2, d2 * chiR2)).reshape(r, d2, chiR2)
    # Ensure last tensor right bond = 1
    return A

#  Build left and right environments
# L[i] has shape (wL_i, chi_i, chi_i) = env to the LEFT of site i
# R[i] has shape (wR_i, chi_i, chi_i) = env to the RIGHT of site i-1
# i runs from 0..N, with L[0]=[[[1]]], R[N]=[[[1]]]


def build_left_envs(A: List[np.ndarray], W: List[np.ndarray]) -> List[np.ndarray]:
    """
    
    Build left environments L:
        L[i] has shape (wL_i, chi_{i}, chi_{i}),
    i.e., the leading dimension matches the left MPO bond of site i while
    the other two are the MPS bond dimensions to the left of site i.
    Recurrence (from left to right):
        Base (to the left of the first site 0): L[0] = ones((1, 1, 1))
        For i = 0, ..., N-1:
            L[i+1] depends on site i and L[i] via contractions of W[i], A[i], A[i]*, and L[i].
    Input:
        A: List of MPS tensors, length N
        W: List of MPO tensors, length N
    Output:
        L: List of left environments, length N+1   
    """
    N = len(A)
    L = [np.ones((1, 1, 1))]
    for i in range(N):
        Li = L[-1]
        Ai = A[i]
        Wi = W[i]
        # To build L[i+1] from L[i] we want to do
        # L[i+1]_{al', b, be'} = sum_{al, s, s', be} L[i]_{al, b, be} * W[i]_{al, al', s, s'} * A[i]_{be, s, chi} * A[i]^*_{be', s', chi}
        t = np.tensordot(Li, Wi, axes=([0], [0]))  # First do sum_al L[i] * W[i]. Now t has shape (b, be, al', s, s')
        t = np.tensordot(t, Ai, axes=([0, 3], [0, 1]))  # Now contract b and s with A[i]'s corr bonds. From prev line we see we need the updated t's 0 and 3 axes. 
        t = np.tensordot(t, Ai.conj(), axes=([0, 2], [0, 1]))  # Finally contract be and s' with A[i]^*. From prev line we see we need the updated t's 0 and 2 axes.
        L.append(t)
    return L  # length N+1


def build_right_envs(A: List[np.ndarray], W: List[np.ndarray]) -> List[np.ndarray]:
    """
    Build right environments R:
      R[i] has shape (wR_i, chi_{i+1}, chi_{i+1}),
    i.e., the leading dimension matches the right MPO bond of site i while
    the other two are the MPS bond dimensions to the right of site i.

    Recurrence (from right to left):
      Base (to the right of the last site N-1): R[N-1] = ones((wR_{N-1}, 1, 1))
      For i = N-2, ..., 0:
        R[i] depends on site i+1 and R[i+1] via contractions of W[i+1], A[i+1], A[i+1]*, and R[i+1].

    This shape matches make_heff_matvec(...), which contracts wr_{i+1}
    with the leading index of R[i+1].
    """
    N = len(A)
    R: List[np.ndarray] = [None] * N

    # Base: to the right of the last site
    wR_last = W[N - 1].shape[1]
    chi_after_last = A[N - 1].shape[2]  # = 1 for OBC
    R[N - 1] = np.ones((wR_last, chi_after_last, chi_after_last))

    # Recurrence
    for i in reversed(range(N - 1)):
        Ai1 = A[i + 1]               # (chi_i, d, chi_{i+1})
        Wi1 = W[i + 1]               # (wL_{i+1}, wR_{i+1}, d, d)
        Ri1 = R[i + 1]               # (wR_{i+1}, chi_{i+1}, chi_{i+1})
        # Contract wR_{i+1}
        t = np.tensordot(Wi1, Ri1, axes=([1], [0]))      # -> (wL_{i+1}, s, s', chi_{i+1}, chi_{i+1})
        # Contract s and chi_{i+1} with A[i+1]
        t = np.tensordot(t, Ai1, axes=([1, 3], [1, 2]))  # -> (wL_{i+1}, s', chi_i)
        # Contract s' and chi_{i+1} with A[i+1]^*
        t = np.tensordot(t, Ai1.conj(), axes=([1, 2], [1, 2]))  # -> (wL_{i+1}, chi_i, chi_i)
        # Identify wL_{i+1} with wR_i
        R[i] = t
        # if (verbose):
        #     for n in range(1,N-1):
        #         print(f"Recursion for R: R[{n}] shape: {R[n].shape if R[n] is not None else None}")



    return R


# Dense two-site effective Hamiltonian builder
# Heff has indices (l,s1,s2,r ; l',s1',s2',r')


def build_heff_dense(Li: np.ndarray, Wi: np.ndarray, Wi1: np.ndarray, Ri1: np.ndarray) -> np.ndarray:
    """
    Li: (wL_i, chiL, chiL)
    Wi: (wL_i, wR_i, d, d)
    Wi1: (wL_{i+1}=wR_i, wR_{i+1}, d, d)
    Ri1: (wR_{i+1}, chiR, chiR)
    Returns dense matrix Heff of shape (chiL*d*d*chiR, chiL*d*d*chiR)
    """
    # Contract left env with W[i]
    t = np.tensordot(Li, Wi, axes=([0], [0]))  # (chiL, chiL, wR_i, s1, s1')
    # Connect to W[i+1] via internal MPO bond (wR_i == wL_{i+1})
    t = np.tensordot(t, Wi1, axes=([2], [0]))  # (chiL, chiL, s1, s1', wR_{i+1}, s2, s2')
    # Contract with right env
    t = np.tensordot(t, Ri1, axes=([4], [0]))  # (chiL, chiL, s1, s1', s2, s2', chiR, chiR)
    # Reorder to group bra and ket indices (l, s1, s2, r ; l', s1', s2', r')
    t = np.transpose(t, (0, 2, 4, 6, 1, 3, 5, 7))
    chiL = Li.shape[1]
    d = Wi.shape[2]
    chiR = Ri1.shape[1]
    Heff = t.reshape(chiL * d * d * chiR, chiL * d * d * chiR)
    # Symmetrize small numerical asymmetries
    Heff = 0.5 * (Heff + Heff.T)
    return Heff



# Expectation value of an MPO with an MPS (scalar)


def expect_mpo(A: List[np.ndarray], W: List[np.ndarray]) -> float:
    L = np.ones((1, 1, 1))
    for i in range(len(A)):
        Ai, Wi = A[i], W[i]
        t = np.tensordot(L, Wi, axes=([0], [0]))  # (al, al', b, s, s')
        t = np.tensordot(t, Ai, axes=([0, 3], [0, 1]))  # (al', b, s', be)
        t = np.tensordot(t, Ai.conj(), axes=([0, 2], [0, 1]))  # (b, be, be')
        L = t
    # At the end L should be shape (1,1,1) scalar
    return float(L.reshape(-1)[0].real)


# Build an MPO that is identity everywhere except at `site` where it is O

def local_op_mpo(N: int, O: np.ndarray, site: int) -> List[np.ndarray]:
    d = O.shape[0]
    IdMPO = [np.eye(d).reshape(1, 1, d, d) for _ in range(N)]
    IdMPO[site] = O.reshape(1, 1, d, d)
    return IdMPO


# Identity MPO (for normalization checks)

def identity_mpo(N: int, d: int) -> List[np.ndarray]:
    return [np.eye(d).reshape(1, 1, d, d) for _ in range(N)]


# Simple product-state MPS for sanity checks


def product_state_mps(N: int, vec: np.ndarray) -> List[np.ndarray]:
    vec = np.asarray(vec, dtype=float)
    vec = vec / np.linalg.norm(vec)
    A = [vec.reshape(1, 2, 1).copy() for _ in range(N)]
    return A


# DMRG driver (two-site sweeps)

def dmrg(A: List[np.ndarray], W: List[np.ndarray], m: int, sweeps: int = 4,
         maxiter_eig: int = 200, tol: float = 1e-8) -> Tuple[List[np.ndarray], List[float]]:
    N = len(A)
    d = A[0].shape[1]
    energies = []
    Id = identity_mpo(N, d)
    for sw in range(sweeps):
        # Left-to-right half-sweep
        Rlist = build_right_envs(A, W)
        if (verbose):
            print(f"Right envs after sweep {sw}:", [r.shape for r in Rlist])
        L = np.ones((1, 1, 1))
        for i in range(N - 1):
            chiL = A[i].shape[0]
            chiR = A[i + 1].shape[2]
            Heff = build_heff_dense(L, W[i], W[i + 1], Rlist[i + 1])
            # Dense symmetric eigensolve (small local dimension)
            vals, vecs = np.linalg.eigh(Heff)
            theta = vecs[:, 0].reshape(chiL, d, d, chiR)
            # SVD split (absorb S to the RIGHT, left-canonical A[i])
            M = theta.reshape(chiL * d, d * chiR)
            U, S, Vh = svd(M, full_matrices=False)
            r = min(m, U.shape[1])
            U = U[:, :r]
            S = S[:r]
            Vh = Vh[:r, :]
            A[i] = U.reshape(chiL, d, r)
            A[i + 1] = (np.diag(S) @ Vh).reshape(r, d, chiR)
            # Update left environment to i+1 using updated A[i]
            t = np.tensordot(L, W[i], axes=([0], [0]))
            t = np.tensordot(t, A[i], axes=([0, 3], [0, 1]))
            t = np.tensordot(t, A[i].conj(), axes=([0, 2], [0, 1]))
            L = t
        # Right-to-left half-sweep
        Llist = build_left_envs(A, W)
        R = np.ones((1, 1, 1))
        for i in reversed(range(N - 1)):
            chiL = A[i].shape[0]
            chiR = A[i + 1].shape[2]
            Heff = build_heff_dense(Llist[i], W[i], W[i + 1], R)
            vals, vecs = np.linalg.eigh(Heff)
            theta = vecs[:, 0].reshape(chiL, d, d, chiR)
            # SVD split: make A[i+1] right-orthonormal, absorb S to the LEFT
            M = theta.reshape(chiL * d, d * chiR)
            U, S, Vh = svd(M, full_matrices=False)
            r = min(m, U.shape[1])
            U = U[:, :r]
            S = S[:r]
            Vh = Vh[:r, :]
            A[i + 1] = Vh.reshape(r, d, chiR)  # right-orthonormal on bond r
            A[i] = (U @ np.diag(S)).reshape(chiL, d, r)
            # Update right environment to i using updated A[i+1]
            t = np.tensordot(W[i + 1], R, axes=([1], [0]))
            t = np.tensordot(t, A[i + 1], axes=([1, 3], [1, 2]))
            t = np.tensordot(t, A[i + 1].conj(), axes=([1, 2], [1, 2]))
            R = t
        # Record normalized energy after full sweep
        E = expect_mpo(A, W)
        norm = expect_mpo(A, Id)
        energies.append(E / norm)
    return A, energies

# Exact Diagonalization (ED) for TFIM OBC (sparse)

def tfim_sparse_hamiltonian(N: int, J: float, h: float) -> csr_matrix:
    I, X, Z = spin_ops()
    d = 2
    H = csr_matrix((d ** N, d ** N), dtype=float)
    # On-site transverse field
    for i in range(N):
        op = None
        for j in range(N):
            oj = X if j == i else np.eye(d)
            op = oj if op is None else spkron(op, oj, format='csr')
        H = H - h * op
    # Nearest-neighbor ZZ
    for i in range(N - 1):
        op = None
        for j in range(N):
            if j == i:
                oj = Z
            elif j == i + 1:
                oj = Z
            else:
                oj = np.eye(d)
            op = oj if op is None else spkron(op, oj, format='csr')
        H = H - J * op
    return H.tocsr()


def ed_ground_state(H: csr_matrix) -> Tuple[float, np.ndarray]:
    # Smallest algebraic eigenpair
    vals, vecs = eigsh(H, k=1, which='SA', tol=1e-10)
    E0 = float(vals[0])
    psi0 = vecs[:, 0]
    # Normalize (eigsh already returns normalized vectors)
    return E0, psi0


def ed_expectation(psi: np.ndarray, op_full: csr_matrix) -> float:
    return float((psi.conj().T @ (op_full @ psi)).real)


def ed_local_op(N: int, op: np.ndarray, site: int) -> csr_matrix:
    d = op.shape[0]
    M = None
    for j in range(N):
        oj = op if j == site else np.eye(d)
        M = oj if M is None else spkron(M, oj, format='csr')
    return M.tocsr()


def run_demo(N: int = 10, J: float = 1.0, h: float = 1.0,
             m_list: Tuple[int, ...] = (4, 8, 16, 32), sweeps: int = 4,
             seed: int = 0) -> None:
    print(f"TFIM demo: N={N}, J={J}, h={h}, sweeps={sweeps}")
    # ED reference
    print("Building ED Hamiltonian...")
    H = tfim_sparse_hamiltonian(N, J, h)
    print("Diagonalizing (ED)...")
    E0, psi0 = ed_ground_state(H)
    I, X, Z = spin_ops()
    mX = np.mean([ed_expectation(psi0, ed_local_op(N, X, i)) for i in range(N)])
    mZ = np.mean([ed_expectation(psi0, ed_local_op(N, Z, i)) for i in range(N)])
    print(f"ED:  E0 = {E0:.12f},  <X> = {mX:.8f},  <Z> = {mZ:.8f}")

    # DMRG comparisons
    print("\nDMRG results:")
    W = tfim_mpo(N, J, h)
    header = f"{'m':>4}  {'E':>15}  {'|E-E0|':>12}  {'<X>':>12}  {'err<X>':>10}  {'<Z>':>12}  {'err<Z>':>10}"
    print(header)
    print('-' * len(header))
    for m in m_list:
        A = random_mps(N, d=2, m=m, seed=seed)
        A, E_sweeps = dmrg(A, W, m=m, sweeps=sweeps)
        Id = identity_mpo(N, 2)
        norm = expect_mpo(A, Id)
        E = expect_mpo(A, W) / norm
        # Local averages via simple MPOs (normalize by norm)
        X_av = 0.0
        Z_av = 0.0
        for i in range(N):
            X_mpo = local_op_mpo(N, X, i)
            Z_mpo = local_op_mpo(N, Z, i)
            X_av += expect_mpo(A, X_mpo) / norm
            Z_av += expect_mpo(A, Z_mpo) / norm
        X_av /= N
        Z_av /= N       
        print(f"{m:4d}  {E:15.12f}  {abs(E - E0):12.4e}  {X_av:12.8f}  {abs(X_av - mX):10.2e}  {Z_av:12.8f}  {abs(Z_av - mZ):10.2e}")
        print("\nDone.")
      




if __name__ == "__main__":
    # You can tweak parameters here
    parser = argparse.ArgumentParser(description="DMRG demo for 1D TFIM")
    parser.add_argument('--N', type=int, default=10, help='Number of sites')
    parser.add_argument('--J', type=float, default=1.0, help='Coupling J')
    parser.add_argument('--h', type=float, default=1.0, help='Transverse field h')
    parser.add_argument('--sweeps', type=int, default=4, help='Number of DMRG sweeps')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()
    N = args.N
    J = args.J
    h = args.h
    sweeps = args.sweeps
    verbose = args.verbose
    
    
    run_demo(N=N, J=J, h=h, m_list=(4, 8, 16, 32), sweeps=sweeps, seed=42)
