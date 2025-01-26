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

#define indices martix for state rho_b

omega = np.exp(2j * np.pi / 3)

def generate_bell_states():
    """
    Generate all nine Bell states |Ω_k,l⟩ using the Weyl operators.
    Returns a list of |Ω_k,l⟩ states as 1D arrays.
    """
    basis = np.eye(3)  # Standard computational basis
    omega_states = []  # Store |Ω_k,l⟩ states

    for k in range(3):
        for l in range(3):
            # Construct W_k,l operator
            W_k_l = np.sum([
                omega ** (j * k) * np.outer(basis[j], basis[(j + l) % 3]) for j in range(3)
            ], axis=0)
            # Define |Ω_0,0⟩ = (|00⟩ + |11⟩ + |22⟩) / sqrt(3)
            omega_0_0 = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]) / np.sqrt(3)
            # Generate |Ω_k,l⟩ from |Ω_0,0⟩
            omega_k_l = np.dot(np.kron(W_k_l, np.eye(3)), omega_0_0)
            omega_states.append(omega_k_l)
    return omega_states


#define rho_b

def rhob(x, bell_states):
    
    d_matrix = np.array([
        [2 * x, 0, 1/3 - x],
        [0, x, 1/3 - x],
        [0, 0, 1/3 - x]
    ])
    
    rho_b = sum(
        d_matrix[k, l] * np.outer(bell_states[k * 3 + l], bell_states[k * 3 + l].conj())
        for k in range(3)
        for l in range(3)
    )
    return rho_b


#define Lorentz boost function to get particle in rest frame to frame O

V = np.array([
        [-1, 1j, 0] / np.sqrt(2),
        [0, 0, np.sqrt(2) / np.sqrt(2)],
        [1, 1j, 0] / np.sqrt(2)])

def lorentz_boost_k(momentum, mass = 1, energy = 1):
    
    momentum = np.asarray(momentum, dtype=float)
    
    k0 = momentum[0]
    mom = momentum[1:]
    
    k_outer = np.outer(mom, mom)
    identity_3x3 = np.eye(3)
    
    block = (mass * identity_3x3) + k_outer / (mass + k0)
    
    boost_matrix = np.zeros((4, 4))
    boost_matrix[0, 0] = k0 / mass
    boost_matrix[0, 1:] = mom / mass
    boost_matrix[1:, 0] = mom / mass
    boost_matrix[1:, 1:] = block
    
    return boost_matrix

#define Lorentz boost to boost particle with certain velocity

def lam_boost(e, xi):
    
    # Ensure e is a numpy array and normalize it to be a unit vector
    e = np.asarray(e, dtype=float)
    e = e / np.linalg.norm(e)

    cosh_xi = np.cosh(xi)
    sinh_xi = np.sinh(xi)

    outer_product = np.outer(e, e)

    block = np.eye(3) + (cosh_xi - 1) * outer_product

    # Assemble the full Lorentz boost matrix
    boost_matrix = np.zeros((4, 4))
    boost_matrix[0, 0] = cosh_xi
    boost_matrix[0, 1:] = e * sinh_xi
    boost_matrix[1:, 0] = e * sinh_xi
    boost_matrix[1:, 1:] = block

    return boost_matrix


#define wigner rotation

def wigner(e, xi, momentum, mass = 1, energy = 1):

    lam_k = np.matmul(lam_boost(e, xi),momentum)
    lrz_lam_k = lorentz_boost_k(lam_k)
    
    wigner = np.matmul(np.linalg.inv(lrz_lam_k),np.matmul(lam_boost(e,xi),lorentz_boost_k(momentum)))
    return np.linalg.qr(wigner)[0][1:,1:]

def D(e, xi, momentum, mass = 1, energy = 1):
    return np.matmul(V,np.matmul(wigner(e, xi, momentum),V.conjugate().transpose()))