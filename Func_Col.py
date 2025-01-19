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
        
        B = np.zeros(b*b)
        list = []
        
        for i in range(b):
            for j in range(b):
                list.append(block[j][i])

        for l in range(len(list)):
            B[l] = np.asarray(list)[l]
        
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

import numpy as np

# Define the standard vectors
standard_vectors = [
    np.array([1, 0, 0], dtype=complex),
    np.array([0, 1, 0], dtype=complex),
    np.array([0, 0, 1], dtype=complex),
]
stvec = standard_vectors

# Define the bell_con function
def bell_con(m=0, n=0):
    # Initialize `ten` as a 3x3 matrix of zeros
    ten = np.zeros(9, dtype=complex)
    
    # Define omega as the cube root of unity
    omega = np.exp(2 * np.pi * 1j / 3)

    # Iterate and compute the Bell state
    for k in range(3):
        prod = omega**(n * k) * np.kron(stvec[k], stvec[(k + m) % 3])
        ten += prod  # Accumulate the sum
    
    return ten

#def bell_con(m,n):
#    
#    omega = np.exp(2*np.pi*1j)
#
#    for k in range(3):
#        bell = 1/np.sqrt(3) * np.sum(omega**(n*k)*np.tensordot(stvec[k],stvec[(k+m)%3],axes=0))
#
#    return bell_states