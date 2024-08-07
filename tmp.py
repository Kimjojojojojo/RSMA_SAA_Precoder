import numpy as np
import cvxpy as cp

def vectorization(A):
    # A의 전치 행렬을 만든 후, 열 단위로 펼쳐서 새로운 1차원 배열로 만듭니다.
    flattened = A.T.flatten()

    # 1차원 배열을 (n, 1) 형태의 column vector로 변환합니다.
    return flattened.reshape(-1, 1)

def make_psd(matrix):
    """
    주어진 Hermitian 행렬을 양의 정부호 행렬로 변환합니다.
    """
    # Hermitian 행렬인지 확인
    if not np.allclose(matrix, matrix.conj().T):
        raise ValueError("입력 행렬은 Hermitian 행렬이어야 합니다.")

    # 고유값 분해
    eigvals, eigvecs = np.linalg.eigh(matrix)

    # 모든 고유값을 양수로 변환 (0보다 작은 값을 0으로)
    eigvals[eigvals < 0] = 0

    # PSD 행렬 생성
    psd_matrix = np.dot(eigvecs, np.dot(np.diag(eigvals), eigvecs.conj().T))
    return psd_matrix

M = 4
Q = 2
K = 2

alpha = 0.7
P_t = 10
sigma_e = (P_t) ** (-1*alpha)
sigma_n = 1

H_sample = (np.random.normal(0, 1, (K, M, Q)) + 1j*np.random.normal(0, 1, (K, M, Q))) / np.sqrt(2)
H_error = (np.random.normal(0, sigma_e, (K, M, Q)) + 1j*np.random.normal(0, sigma_e, (K, M, Q))) / np.sqrt(2*sigma_e)
H_hat = H_sample - H_error
H_concat = np.hstack(H_hat)


P_k = np.zeros((K, M, Q), dtype= complex)
P_k_H = np.zeros((K, Q, M), dtype= complex)
# Private precoder P initialization
for k in range(K): # Private precoder
    P_k[k] = H_hat[k]
    P_k_H[k] = np.conjugate(P_k[k]).T

U, Sigma, Vt = np.linalg.svd(H_concat, full_matrices=False)

# common precoder P_c initialization
P_c = U[:, :Q]
P_c_H = np.conjugate(P_c).T


iter_num = 100
G_tmp=[]
U_tmp=[]
P_tmp=[]

A_prime_ck_tmp=[]
A_ck_tmp=[]
A_pk_tmp=[]
a_ck_tmp=[]
a_pk_tmp=[]
Phi_ck_tmp=[]
Phi_pk_tmp=[]

# CVX variables
mu = np.ones(K)
X_hat = cp.Variable(K)
P = [cp.Variable((M, Q), complex=True) for _ in range(K + 1)]  # 0th vector is common signal precoder
P_vector = cp.vstack([cp.vec(Pk) for Pk in P])
P_total = cp.hstack(P)
p_k = np.zeros((K, Q * M, 1), dtype=complex)

print(H_hat)
# AO Algorithm
for n in range(iter_num):
    print(f"{n}th iteration")
    H_tilde = (np.random.normal(0,sigma_e, (K, M, Q)) + 1j*np.random.normal(0,sigma_e, (K, M, Q))) / np.sqrt(2*sigma_e)
    H_k = H_hat + H_tilde
    H_k_H = np.zeros((K, Q, M), dtype= complex)
    #print(H_k)
    for k in range(K): # H^H
        H_k_H[k] = np.conjugate(H_k[k]).T
        eigenvalues = np.linalg.eigvals(H_k[k] @ H_k_H[k])
        for eig in eigenvalues:
            if eig < 0:
                print("U_ck[k] Eigenvalues:", eig)



    # Generate interference covariance matrices
    R_ck = np.zeros((K,Q,Q), dtype= complex)
    R_pk = np.zeros((K,Q,Q), dtype= complex)
    for k in range(K):
        R_ck[k] = np.eye(Q) + np.sum([H_k_H[k] @ P_k[i] @ P_k_H[i] @ H_k[k] for i in range(K)])
        R_pk[k] = R_ck[k] - H_k_H[k] @ P_k[k] @ P_k_H[k] @ H_k[k]


    # Generate MMSE filter matrix
    G_ck = np.zeros((K, Q, Q), dtype= complex)
    G_pk = np.zeros((K, Q, Q), dtype= complex)
    G_ck_H = np.zeros((K, Q, Q), dtype= complex)
    G_pk_H = np.zeros((K, Q, Q), dtype= complex)
    for k in range(K):
        G_ck[k] = P_c_H @ H_k[k] / (H_k_H[k] @ P_c @ P_c_H @ H_k[k] + R_ck[k])
        G_pk[k] = P_k_H[k] @ H_k[k] / (H_k_H[k] @ P_k[k] @ P_k_H[k] @ H_k[k] + R_pk[k])
        G_ck_H[k] = np.conjugate(G_ck[k]).T
        G_pk_H[k] = np.conjugate(G_pk[k]).T

    # Generate MMSE matrix and weight matrix
    E_ck = np.zeros((K, Q, Q), dtype= complex)
    E_pk = np.zeros((K, Q, Q), dtype= complex)
    U_ck = np.zeros((K, Q, Q), dtype= complex)
    U_pk = np.zeros((K, Q, Q), dtype= complex)
    U_ck_H = np.zeros((K, Q, Q), dtype= complex)
    U_pk_H = np.zeros((K, Q, Q), dtype= complex)
    for k in range(K):
        E_ck[k] = np.linalg.inv(np.eye(Q) + P_c_H @ H_k[k] / (R_ck[k]) @ H_k_H[k] @ P_c)
        E_pk[k] = np.linalg.inv(np.eye(Q) + P_k_H[k] @ H_k[k] / (R_pk[k]) @ H_k_H[k] @ P_k[k])
        U_ck[k] = np.eye(Q) + P_c_H @ H_k[k] / (R_ck[k]) @ H_k_H[k] @ P_c
        U_pk[k] = np.eye(Q) + P_k_H[k] @ H_k[k] / (R_pk[k]) @ H_k_H[k] @ P_k[k]
        U_ck_H[k] = np.conjugate(U_ck[k]).T
        U_pk_H[k] = np.conjugate(U_pk[k]).T
        eigenvalues = np.linalg.eigvals(U_ck[k])



    # Vectorization
    p_c = vectorization(P_c)
    for k in range(K):
        p_k[k] = vectorization(P_k[k])

    # Compute A, A', a, a'
    A_prime_ck = np.zeros((K, Q * M, Q * M), dtype=complex)
    A_ck = np.zeros((K, Q * M, Q * M), dtype=complex)
    A_pk = np.zeros((K, Q * M, Q * M), dtype=complex)
    a_ck = np.zeros((K, Q * M, 1), dtype=complex)
    a_pk = np.zeros((K, Q * M, 1), dtype=complex)
    Phi_ck = np.zeros(K)
    Phi_pk = np.zeros(K)

    for k in range(K):
        A_prime_ck[k] = np.kron(np.eye(Q), H_k[k] @ G_ck_H[k] @ U_ck[k] @ G_ck[k] @ H_k_H[k])
        A_ck[k] = np.kron(np.eye(Q), H_k[k] @ G_ck_H[k] @ U_ck[k] @ G_ck[k] @ H_k_H[k])
        A_pk[k] = np.kron(np.eye(Q), H_k[k] @ G_pk_H[k] @ U_pk[k] @ G_pk[k] @ H_k_H[k])
        a_ck[k] = vectorization(H_k[k] @ G_ck_H[k] @ U_ck_H[k])
        a_pk[k] = vectorization(H_k[k] @ G_pk_H[k] @ U_pk_H[k])
        Phi_ck[k] = np.real(
            1 * np.trace(U_ck[k] @ G_ck[k] @ G_ck_H[k]) + np.trace(U_ck[k]) - np.log2(np.linalg.det(U_ck[k])))
        Phi_pk[k] = np.real(
            1 * np.trace(U_pk[k] @ G_pk[k] @ G_pk_H[k]) + np.trace(U_pk[k]) - np.log2(np.linalg.det(U_pk[k])))

    A_prime_ck_tmp.append(A_prime_ck)
    A_ck_tmp.append(A_ck)
    A_pk_tmp.append(A_pk)
    a_ck_tmp.append(a_ck)
    a_pk_tmp.append(a_pk)
    Phi_ck_tmp.append(Phi_ck)
    Phi_pk_tmp.append(Phi_pk)

    A_hat_prime_ck = np.mean(A_prime_ck_tmp, 0)
    A_hat_ck = np.mean(A_ck_tmp, 0)
    A_hat_pk = np.mean(A_pk_tmp, 0)
    a_hat_ck = np.mean(a_ck_tmp, 0)
    a_hat_pk = np.mean(a_pk_tmp, 0)
    Phi_hat_ck = np.mean(Phi_ck_tmp, 0)
    Phi_hat_pk = np.mean(Phi_pk_tmp, 0)

    ### Obtain precoder P through CVX tool ###
    objective_expr = 0
    for k in range(1, K + 1):
        A_hat_pk_symmetric = 0.5 * (A_hat_pk[k - 1] + A_hat_pk[k - 1].conj().T)
        A_hat_pk_symmetric_psd = make_psd(A_hat_pk_symmetric)

        term1 = X_hat[k - 1]
        term2 = cp.real(cp.quad_form(P_vector[k], A_hat_pk_symmetric_psd))
        term3 = 0
        for i in range(1, K + 1):
            if i != k:
                term3 += cp.quad_form(P_vector[i], A_hat_pk_symmetric_psd)
        terms3 = cp.real(term3)
        term4 = -2 * cp.real(a_hat_pk[k - 1].conj().T @ P_vector[k])
        term5 = Phi_hat_pk[k - 1]

        objective_expr += mu[k - 1] * (term1 + term2 + term3 + term4 + term5)

    objective = cp.Minimize(objective_expr)

    constraints = []

    for k in range(1, K + 1):
        A_hat_prime_ck_symmetric = 0.5 * (A_hat_prime_ck[k - 1] + A_hat_prime_ck[k - 1].conj().T)
        A_hat_prime_ck_symmetric_psd = make_psd(A_hat_prime_ck_symmetric)
        A_hat_ck_symmetric = 0.5 * (A_hat_ck[k - 1] + A_hat_ck[k - 1].conj().T)
        A_hat_ck_symmetric_psd = make_psd(A_hat_ck_symmetric)

        lhs = cp.sum(X_hat, axis=0) + Q
        rhs1 = cp.quad_form(P_vector[0], A_hat_prime_ck_symmetric_psd)
        rhs2 = 0
        for i in range(1, K + 1):
            rhs2 += cp.quad_form(P_vector[i], A_hat_ck_symmetric_psd)
        rhs3 = - 2 * cp.real(a_hat_ck[k - 1].conj().T @ P_vector[0]) + Phi_ck[k - 1]
        rhs = rhs1 + rhs2 + rhs3
        constraints.append(lhs >= rhs)

    constraints.append(cp.norm(P_total, 'fro') ** 2 <= P_t)
    for k in range(K):
        constraints.append(X_hat[k] <= 0)

    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    print("P:", result)
    if result is not None:
        P_values = [P_k.value for P_k in P]
        X_hat_values = X_hat.value

        for k in range(K + 1):
            if k == 0:
                P_c = P_values[k]
                P_c_H = np.conjugate(P_c).T
            else:
                P_k[k - 1] = P_values[k]
                P_k_H[k - 1] = np.conjugate(P_k[k - 1]).T
    else:
        print("최적화 문제에 대한 해를 찾을 수 없습니다.")

    # print("최적 값:", result)
    # for k in range(K + 1):
    #     print(f"최적 P[{k}]:\n", P[k].value)
