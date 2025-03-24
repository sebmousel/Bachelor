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
    
    # If all eigenvalues are negative, the state is PPT
    return np.all(eigenvalues >= 0)
    

#return dimension of matrix

def get_dim(matrix):
    return np.shape(matrix)


#divide matrix into blocks

def blck(matrix,blocksize = 3):
    if matrix is None:
        raise ValueError("Error: matrix is None in realign_log() before calling SVD")
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


#get singular values of matrix

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
    

#define normalisation

def nrm(vector):
    return vector / np.linalg.norm(vector)


#construction of bell states

#define indices martix for state rho_b and omega

omega = np.exp(2j * np.pi / 3)

def generate_bell_states():
    basis = np.eye(3)
    omega_states = []

    for k in range(3):
        for l in range(3):
            W_k_l = np.sum([
                omega ** (j * k) * np.outer(basis[j], basis[(j + l) % 3]) for j in range(3)
            ], axis=0)
            omega_0_0 = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]) / np.sqrt(3)
            omega_k_l = np.dot(np.kron(W_k_l, np.eye(3)), omega_0_0)
            omega_states.append(nrm(omega_k_l))
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

def rho2(bell_states):

    a_matrix = np.array([
        [0,2/9,2/9],
        [0,2/9,2/9],
        [5/18,0,0]
    ])

    rho_2 = sum(
        a_matrix[k,l] * np.outer(bell_states[k * 3 + l], bell_states[k * 3 + l].conj())
        for k in range(3)
        for l in range(3)
    )
    return rho_2


#define Lorentz boost function to get particle in rest frame to frame O and matrix V

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
    
    e = np.asarray(e, dtype=float)
    e = e / np.linalg.norm(e)

    if np.isscalar(xi):
        cosh_xi = np.cosh(xi)
        sinh_xi = np.sinh(xi)
    else:
        raise ValueError(f"xi must be a scalar value. xi is rn {xi}")

    outer_product = np.outer(e, e)

    block = np.eye(3) + (cosh_xi - 1) * outer_product

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


#define the 3dim D matrix

def D(e, xi, momentum, mass = 1, energy = 1):
    return np.matmul(V,np.matmul(wigner(e, xi, momentum),V.conjugate().transpose()))


#define the values for realignment plot

def boosted_state1(e, xi, mom1, mom2,n):

    Dmat = [None,
            D(e,xi,mom1),
            D(e,xi,mom2)]

    sum = np.zeros_like(rhob(n,generate_bell_states()),dtype=complex)

    for i,j in [(1,2),(2,1)]:
        sum += (1/2) * np.matmul(
                np.kron(Dmat[i].T,Dmat[j].T),np.matmul(rhob(n,generate_bell_states()),np.kron(Dmat[i].conj().T,Dmat[j].conj().T))
                        )
                
    return sum

def realign_val_12_21(e, xi, mom1, mom2):

    val = []
    PPT = []

    for n in np.linspace(0, 1/3 , 1000):
        
        sum = boosted_state1(e,xi,mom1,mom2,n)
        val.append(realign_log(sum/np.trace(sum)))
        PPT.append(is_PPT(sum))
    return val,PPT


#define values for values of realignment with momenta entanglement dependant of theta version 2

def boosted_state2(e, xi, mom1, mom2,theta):

    Dmat = [None,
            D(e,xi,mom1),
            D(e,xi,mom2)]
    
    matrix = []

    for n in np.linspace(0, 1/3 , 1000):

        sum = np.zeros_like(rhob(n,generate_bell_states()),dtype=complex)

        for i,j in [(1,2),(2,1)]:
            if (i,j) == (1,2):
                sum += np.cos(theta)**2 * np.matmul(
                        np.kron(Dmat[i].T,Dmat[j].T),np.matmul(rhob(n,generate_bell_states()),np.kron(Dmat[i].conj().T,Dmat[j].conj().T))
                        )
            else:
                sum += np.sin(theta)**2 * np.matmul(
                        np.kron(Dmat[i].T,Dmat[j].T),np.matmul(rhob(n,generate_bell_states()),np.kron(Dmat[i].conj().T,Dmat[j].conj().T))
                        )
                
        matrix.append(sum)

    return matrix

def realign_val_theta2(e,xi,mom1,mom2,theta):

    val = []

    for m in boosted_state2(e,xi,mom1,mom2,theta):                
        val.append(realign_log(m))
    
    return val

def realign_general(e,xi,mom1,mom2,rho1,rho2,theta1,theta2,p):
    
    Dmat = [None,
            D(e,xi,mom1),
            D(e,xi,mom2)]
    
    sum1 = np.zeros_like(rhob(0.1,generate_bell_states()),dtype=complex)
    sum2 = np.zeros_like(rhob(0.1,generate_bell_states()),dtype=complex)
    
    for i,j in [(1,2),(2,1)]:
            if (i,j) == (1,2):
                sum1 += np.cos(theta1)**2 * np.matmul(
                        np.kron(Dmat[i].T,Dmat[j].T),np.matmul(rho1,np.kron(Dmat[i].conj().T,Dmat[j].conj().T)))
            else:
                sum1 += np.sin(theta1)**2 * np.matmul(
                        np.kron(Dmat[i].T,Dmat[j].T),np.matmul(rho1,np.kron(Dmat[i].conj().T,Dmat[j].conj().T)))
    
    for i,j in [(1,2),(2,1)]:
            if (i,j) == (1,2):
                sum2 += np.matmul(
                        np.kron(Dmat[i].T,Dmat[j].T),np.matmul(rho2,np.kron(Dmat[i].conj().T,Dmat[j].conj().T)))

    sum = p*sum1 + (1-p)*sum2
    
    return realign_log(sum/np.trace(sum)), is_PPT(sum/np.trace(sum)), sum