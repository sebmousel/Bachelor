import numpy as np
import scipy.linalg as la
from scipy.linalg import svd

#partial transpose of 9x9 density matrix

def parttran(matrix, subsystem_dim = 3):
    return matrix.reshape(3,3,3,3).transpose(0,2,1,3).transpose(0,1,3,2).transpose(0,2,1,3).reshape(9,9)


#check Perez-Horodecki-Criterion

def is_PPT(matrix, subsystem_dims = 3):

    rho_partial_transpose = parttran(matrix)
    
    # Compute the eigenvalues of the partially transposed matrix
    eigenvalues = la.eigvalsh(rho_partial_transpose)
    
    # If any eigenvalue is negative, the state is entangled
    if np.any(eigenvalues < 0):
        return True  # Entangled
    else:
        return False  # Separable (Free)
    

#to find trace, use np.trace

def purity(matrix):
    return np.trace(np.matmul(matrix, matrix))


#get dimension

def get_dim(matrix):
    return np.shape(matrix)


#get realignment criterion
#
#create blocks from a matrix

def blck(matrix,blocksize = 3):
    blocks = []
    b = blocksize
    for i in range(b):
        for j in range(b):
            blocks.append(matrix[i*b:b+i*b,j*b:b+j*b])
    return blocks


#realign matrix

def realign(matrix, blocksize = 3):
    
    blocks = blck(matrix)
    b = blocksize
    newblocks = np.empty(0)
    
    for block in blocks:
        
        B = np.zeros(b*b,dtype = np.complex128)
        list = []
        
        for i in range(b):
            for j in range(b):
                list.append(block[j][i])

        for l in range(len(list)):
            B[l] = np.asarray(list, dtype = np.complex128)[l]
        
        newblocks = np.append(newblocks, B)
    
    return newblocks.reshape((9,9))


#get singular values

def SVD(matrix):
    bl = realign(matrix)
    a,b,c = svd(bl)
    return b


#calculate singular criterion

def realign_log(matrix):
    arr = SVD(matrix)
    return np.log2(np.sum(arr))


#evaluate realignment criterion

def realign_crit(matrix):
    val = realign_log
    if val > 0:
        return True
    else:
        return False
    

#construction of bell states

# Define construction of bell states function

d = 3  # Dimension for qutrits
omega = np.exp(2j * np.pi / d)  # Primitive root of unity

# Define the basis states |0⟩, |1⟩, |2⟩
basis_states = np.eye(d)

# Define the Weyl operator W_{k,l}
def weyl_operator(k, l, d=3):
    W = np.zeros((d, d), dtype=complex)
    for j in range(d):
        W += (omega**(j * k)) * np.outer(basis_states[j], basis_states[(j + l) % d])
    return W

# Define the reference state |Ω0,0⟩ = (|00⟩ + |11⟩ + |22⟩)/√3
Omega_00 = (1/np.sqrt(3)) * (np.kron(basis_states[0], basis_states[0]) + 
                             np.kron(basis_states[1], basis_states[1]) + 
                             np.kron(basis_states[2], basis_states[2]))

# Generate the entangled states |Ω_{k,l}⟩ = W_{k,l} ⊗ I_3 |Ω_{0,0}⟩
def generate_entangled_state(k, l, d=3):
    W_kl = weyl_operator(k, l, d)
    W_kl_kron = np.kron(W_kl, np.eye(d))  # Apply W_{k,l} on the first system
    return W_kl_kron @ Omega_00

def gen_bell(d=3):

    bell_states = []

    for k in range(d):
        for l in range(d):
            bell_states.append(generate_entangled_state(k,l).reshape(3,3))

    return bell_states

#define indices martix for state rho_b

def dkl(x):
    d = np.array([
        [2*x, 0,1/3- x],
        [0,x,1/3 - x],
        [0,0,1/3 - x]
    ])
    return d


#define rho_b

def rhob(x):

    bell_states = gen_bell()

    sum = np.zeros((3,3),dtype=np.complex128)

    for m in range(3):
        for n in range(3):
            sum += dkl(x)[m][n]*bell_states[3*m+n]

    return np.outer(sum,sum.conjugate())

#define Lorentz boost function to get particle in rest frame to frame O

V = np.array([
        [-1, 1j, 0],
        [0, 0, np.sqrt(2)],
        [1, 1j, 0]])

import numpy as np

def lorentz_boost_k(momentum, mass = 1, energy = 1):
    """
    Compute the Lorentz boost matrix for a given 3-momentum and mass.

    Parameters:
        momentum (numpy.ndarray): A 3-element array representing the spatial momentum (px, py, pz).
        mass (float): The mass of the particle.

    Returns:
        numpy.ndarray: A 4x4 Lorentz transformation matrix.
    """
    # Ensure momentum is a numpy array
    momentum = np.asarray(momentum, dtype=float)
    
    # Compute energy component k0 (assuming natural units where c=1)
    k0 = momentum[0]
    mom = momentum[1:]
    
    # Construct the 3x3 spatial part of the boost matrix
    k_outer = np.outer(mom, mom)  # Outer product: k ⊗ k^T
    identity_3x3 = np.eye(3)  # 3x3 identity matrix
    
    # Compute the spatial block of the matrix
    spatial_block = (mass * identity_3x3) + k_outer / (mass + k0)
    
    # Assemble the full Lorentz boost matrix
    boost_matrix = np.zeros((4, 4))
    boost_matrix[0, 0] = k0 / mass
    boost_matrix[0, 1:] = mom / mass  # First row (excluding time component)
    boost_matrix[1:, 0] = mom / mass  # First column (excluding time component)
    boost_matrix[1:, 1:] = spatial_block
    
    return boost_matrix

#define Lorentz boost to boost particle with certain velocity

def lam_boost(e, xi):
    """
    Compute the Lorentz boost matrix Λ(e, ξ) given a unit direction vector e and rapidity ξ.

    Parameters:
        e (numpy.ndarray): A 3-element unit vector representing the boost direction.
        xi (float): The rapidity of the boost.

    Returns:
        numpy.ndarray: A 4x4 Lorentz transformation matrix.
    """
    # Ensure e is a numpy array and normalize it to be a unit vector
    e = np.asarray(e, dtype=float)
    e = e / np.linalg.norm(e)

    # Compute hyperbolic functions of rapidity
    cosh_xi = np.cosh(xi)
    sinh_xi = np.sinh(xi)

    # Compute the outer product of e (e ⊗ e^T)
    outer_product = np.outer(e, e)

    # Construct the spatial part of the matrix
    spatial_block = np.eye(3) + (cosh_xi - 1) * outer_product

    # Assemble the full Lorentz boost matrix
    boost_matrix = np.zeros((4, 4))
    boost_matrix[0, 0] = cosh_xi
    boost_matrix[0, 1:] = e * sinh_xi  # First row (excluding time component)
    boost_matrix[1:, 0] = e * sinh_xi  # First column (excluding time component)
    boost_matrix[1:, 1:] = spatial_block

    return boost_matrix


#define wigner rotation

def wigner(e, xi, momentum, mass = 1, energy = 1):

    step1 = np.matmul(lam_boost(e, xi),momentum)
    step2 = lorentz_boost_k(step1)
    step3 = np.matmul(lam_boost(e,xi),lorentz_boost_k(momentum))
    
    return np.matmul(np.linalg.inv(step2),step3)

def D(e, xi, momentum, mass = 1, energy = 1):
    return np.matmul(V / np.sqrt(2),np.matmul(wigner(e, xi, momentum, mass = 1, energy = 1),V.conjugate().transpose() / np.sqrt(2)))