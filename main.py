import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

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


def generate_gaussian_channel(K, M, Q, sigma_e):
    H = np.zeros((K, M, Q), dtype=complex)
    for k in range(K):
        real_part = np.random.normal(0, sigma_e, (M, Q))
        imag_part = np.random.normal(0, sigma_e, (M, Q))
        H[k] = (real_part + 1j * imag_part) / np.sqrt(2 * sigma_e)

    threshold = 10^(-5)
    for k in range(K):
        H_kq = H[k]
        # SVD
        U, S, Vh = np.linalg.svd(H_kq, full_matrices=False)
        # Thresholding
        S[S < threshold] = 0
        # Reconstruct matrix
        H[k] = U @ np.diag(S) @ Vh

    return H


M = 4
Q = 2
K = 2

alpha = 1
P_t = 10
sigma_e = (P_t) ** (-1*alpha)
sigma_n = 1
sample_num = 1000

H_true = generate_gaussian_channel(K, M, Q, 1)
H_error = generate_gaussian_channel(K, M, Q, sigma_e)
H_hat = H_true - H_error
H_tmp = []
H_H_tmp = []
for s in range(sample_num):
    H_tilde = generate_gaussian_channel(K, M, Q, sigma_e)
    H = H_hat + H_tilde
    H_tmp.append(H)
    H_H = np.zeros((K, Q, M), dtype=complex)
    # print(H_k)
    for k in range(K):  # H^H
        H_H[k] = np.conjugate(H[k]).T
    H_H_tmp.append(H_H)

H_concat = np.hstack(H_hat)

# Private precoder P_k initialization
P_k = np.zeros((K, M, Q), dtype= complex)
P_k_H = np.zeros((K, Q, M), dtype= complex)
for k in range(K): # Private precoder
    P_k[k] = H_hat[k]
    P_k_H[k] = np.conjugate(P_k[k]).T

U, Sigma, Vt = np.linalg.svd(H_concat, full_matrices=False)

# common precoder P_c initialization
P_c = U[:, :Q]
P_c_H = np.conjugate(P_c).T


iter_num = 100


# Interference term
R_ck = np.zeros((K, Q, Q), dtype=complex)
R_pk = np.zeros((K, Q, Q), dtype=complex)

# CVX variables
mu = np.ones(K)
X_hat = cp.Variable(K)
x_hat = np.zeros(K)
p_k_cp = [cp.Variable((M * Q, 1), complex=True) for _ in range(K)]  # 0th vector is common signal precoder
p_c_cp = cp.Variable((M * Q, 1), complex=True)
p_k = np.zeros((K, Q * M, 1), dtype=complex)
p_c = np.zeros((Q*M, 1), dtype=complex)

sum_data_rate_list = []
# AO Algorithm
for n in range(iter_num):
    print(f"{n+1}th iteration")
    # Initialize tmp array
    A_prime_ck_tmp = []
    A_ck_tmp = []
    A_pk_tmp = []
    a_ck_tmp = []
    a_pk_tmp = []
    Phi_ck_tmp = []
    Phi_pk_tmp = []
    for s in range(sample_num):
        H_k = H_tmp[s]
        H_k_H = H_H_tmp[s]

        # Generate interference covariance matrices
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
            a_ck[k] = vectorization(H_k[k] @ G_ck_H[k] @ U_ck[k])
            a_pk[k] = vectorization(H_k[k] @ G_pk_H[k] @ U_pk[k])
            Phi_ck[k] = np.real(
                sigma_n * np.trace(U_ck[k] @ G_ck[k] @ G_ck_H[k]) + np.trace(U_ck[k]) - np.log2(np.linalg.det(U_ck[k])))
            Phi_pk[k] = np.real(
                sigma_n * np.trace(U_pk[k] @ G_pk[k] @ G_pk_H[k]) + np.trace(U_pk[k]) - np.log2(np.linalg.det(U_pk[k])))

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
    for k in range(K):
        A_hat_pk_symmetric = 0.5 * (A_hat_pk[k] + A_hat_pk[k].conj().T)
        A_hat_pk_symmetric_psd = make_psd(A_hat_pk_symmetric)
        #print(A_hat_pk_symmetric)
        term1 = X_hat[k]
        term2 = cp.real(cp.quad_form(p_k_cp[k], A_hat_pk_symmetric_psd))
        term3 = 0
        for i in range(K):
            if i != k:
                term3 += cp.quad_form(p_k_cp[i], A_hat_pk_symmetric_psd)
        terms3 = cp.real(term3)
        term4 = -2 * cp.real(a_hat_pk[k].conj().T @ p_k_cp[k])
        term5 = Phi_hat_pk[k]

        objective_expr += mu[k] * (term1 + term2 + terms3 + term4 + term5)

    objective = cp.Minimize(objective_expr)

    constraints = []

    # 1st constraint
    for k in range(K):
        A_hat_prime_ck_symmetric = 0.5 * (A_hat_prime_ck[k] + A_hat_prime_ck[k].conj().T)
        A_hat_ck_symmetric = 0.5 * (A_hat_ck[k] + A_hat_ck[k].conj().T)

        A_hat_prime_ck_symmetric_psd = make_psd(A_hat_prime_ck_symmetric)
        A_hat_ck_symmetric_psd = make_psd(A_hat_ck_symmetric)

        lhs = cp.sum(X_hat, axis=0) + Q
        rhs1 = cp.quad_form(p_c_cp, A_hat_prime_ck_symmetric_psd)
        rhs2 = 0
        for i in range(K):
            rhs2 += cp.quad_form(p_k_cp[i], A_hat_ck_symmetric_psd)
        rhs3 = - 2 * cp.real(a_hat_ck[k].conj().T @ p_c_cp) + Phi_hat_ck[k]
        rhs = rhs1 + rhs2 + rhs3
        constraints.append(lhs >= rhs)

    # 2nd constraint
    power_constraints_common = cp.sum_squares(p_c_cp)
    power_constraints_private = cp.sum([cp.sum_squares(p_k_cp[k]) for k in range(K)])
    power_constraints_sum = power_constraints_private + power_constraints_common
    constraints.append(power_constraints_sum <= P_t)

    # 3rd constraint
    for k in range(K):
        constraints.append(X_hat[k] <= 0)

    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver=cp.SCS)

    # Return results
    X_hat_values = X_hat.value
    x_hat = X_hat_values
    for k in range(K):
        p_k[k] = p_k_cp[k].value
    p_c = p_c_cp.value
    print(p_c)
    power_common = np.linalg.norm(p_c)**2
    power_private = np.sum([np.linalg.norm(p_k[k])**2 for k in range(K)])
    power = power_common + power_private
    print("power_common:", power_common)
    print("power_private:", power_private)
    print("total power:",power)

    for k in range(K):
        P_k[k] = p_k[k].reshape(P_k[0].shape)
        P_k_H[k] = np.conjugate(P_k[k]).T

    # Vector to matrix
    P_c = p_c.reshape(P_c.shape)
    P_c_H = np.conjugate(P_c).T


# Calculate data rate
    r_ck = np.zeros(K)
    r_pk = np.zeros(K)

    R_ck_true = np.zeros((K, Q, Q), dtype=complex)
    R_pk_true = np.zeros((K, Q, Q), dtype=complex)

    for k in range(K):
        R_ck_true[k] = np.eye(Q) + np.sum([H_true[k].conj().T[k] @ P_k[i] @ P_k_H[i] @ H_true[k] for i in range(K)])
        R_pk_true[k] = R_ck[k] - H_true[k].conj().T @ P_k[k] @ P_k_H[k] @ H_true[k]

    for k in range(K):
        r_ck[k] = np.real(np.log2(np.linalg.det(np.eye(Q) + P_c_H @ H_true[k] / (R_ck_true[k]) @ H_true[k].conj().T @ P_c)))
        r_pk[k] = np.real(np.log2(np.linalg.det(np.eye(Q) + P_k_H[k] @ H_true[k] / (R_pk_true[k]) @ H_true[k].conj().T @ P_k[k])))

    sum_data_rate = np.sum(r_pk) + np.min(r_ck)
    sum_data_rate_list.append(sum_data_rate)

    # Iteration의 진행상황 출력
    print(f"Iteration {n + 1}: Sum Data Rate = {sum_data_rate}")

plt.figure(figsize=(10, 6))
plt.plot(range(1, iter_num + 1), sum_data_rate_list, marker='o', linestyle='-', color='b')
plt.title('Sum Data Rate over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Sum Data Rate')
plt.grid(True)
plt.show()