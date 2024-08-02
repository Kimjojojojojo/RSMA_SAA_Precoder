import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

def vectorization(A):
    flattened = A.T.flatten()
    return flattened.reshape(-1, 1)

M = 4
Q = 2
K = 2

alpha = 0.7
P_t = 10
sigma_e = (P_t) ** (-1*alpha)
sigma_n = 1

H_sample = np.random.normal(0, 1, (K, M, Q))
H_error = np.random.normal(0, sigma_e, (K, M, Q))
H_hat = H_sample - H_error
H_concat = np.hstack(H_hat)

P_k = np.zeros((K, M, Q))
P_k_H = np.zeros((K, Q, M))
for k in range(K):
    P_k[k] = H_hat[k]
    P_k_H[k] = np.conjugate(H_hat[k]).T

U, Sigma, Vt = np.linalg.svd(H_concat, full_matrices=False)
P_c = U[:, :Q]
P_c_H = np.conjugate(P_c).T

iter_num = 100
G_tmp = []
U_tmp = []
P_tmp = []

A_prime_ck_tmp = []
A_ck_tmp = []
A_pk_tmp = []
a_ck_tmp = []
a_pk_tmp = []
Phi_ck_tmp = []
Phi_pk_tmp = []

error_norms = []

P_prev = [np.zeros((M, Q), dtype=complex) for _ in range(K + 1)]

for n in range(iter_num):
    print(f"{n}th iteration")
    H_tilde = np.random.normal(0, sigma_e, (K, M, Q))
    H_k = H_hat + H_tilde
    H_k_H = np.zeros((K, Q, M))
    for k in range(K):
        H_k_H[k] = np.conjugate(H_k[k]).T

    for k in range(K):
        P_k_H[k] = np.conjugate(P_k[k]).T

    R_ck = np.zeros((K, Q, Q))
    R_pk = np.zeros((K, Q, Q))
    for k in range(K):
        R_ck[k] = np.eye(Q) + np.sum([H_k_H[k] @ P_k[i] @ P_k_H[i] @ H_k[k] for i in range(K)])
        R_pk[k] = R_ck[k] - H_k_H[k] @ P_k[k] @ P_k_H[k] @ H_k[k]

    G_ck = np.zeros((K, Q, Q))
    G_pk = np.zeros((K, Q, Q))
    G_ck_H = np.zeros((K, Q, Q))
    G_pk_H = np.zeros((K, Q, Q))
    for k in range(K):
        G_ck[k] = P_c_H @ H_k[k] / (H_k_H[k] @ P_c @ P_c_H @ H_k[k] + R_ck[k])
        G_pk[k] = P_k_H[k] @ H_k[k] / (H_k_H[k] @ P_k[k] @ P_k_H[k] @ H_k[k] + R_pk[k])
        G_ck_H[k] = np.conjugate(G_ck[k]).T
        G_pk_H[k] = np.conjugate(G_pk[k]).T

    E_ck = np.zeros((K, Q, Q))
    E_pk = np.zeros((K, Q, Q))
    U_ck = np.zeros((K, Q, Q))
    U_pk = np.zeros((K, Q, Q))
    U_ck_H = np.zeros((K, Q, Q))
    U_pk_H = np.zeros((K, Q, Q))
    for k in range(K):
        E_ck[k] = np.eye(Q) / (np.eye(Q) + P_c_H @ H_k[k] / (R_ck[k]) @ H_k_H[k] @ P_c)
        E_pk[k] = np.eye(Q) / (np.eye(Q) + P_k_H[k] @ H_k[k] / (R_pk[k]) @ H_k_H[k] @ P_k[k])
        U_ck[k] = np.eye(Q) + P_c_H @ H_k[k] / R_ck[k] @ H_k_H[k] @ P_c
        U_pk[k] = np.eye(Q) + P_k_H[k] @ H_k[k] / R_pk[k] @ H_k_H[k] @ P_k[k]
        U_ck_H[k] = np.conjugate(U_ck[k]).T
        U_pk_H[k] = np.conjugate(U_pk[k]).T

    p_c = vectorization(P_c)
    p_k = np.zeros((K, Q*M, 1))
    for k in range(K):
        p_k[k] = vectorization(P_k[k])

    A_prime_ck = np.zeros((K, Q*M, Q*M))
    A_ck = np.zeros((K, Q*M, Q*M))
    A_pk = np.zeros((K, Q*M, Q*M))
    a_ck = np.zeros((K, Q*M, 1))
    a_pk = np.zeros((K, Q*M, 1))
    Phi_ck = np.zeros(K)
    Phi_pk = np.zeros(K)

    for k in range(K):
        A_prime_ck[k] = np.kron(np.eye(Q), H_k[k] @ G_ck_H[k] @ U_ck[k] @ G_ck[k] @ H_k_H[k])
        A_ck[k] = np.kron(np.eye(Q), H_k[k] @ G_ck_H[k] @ U_ck[k] @ G_ck[k] @ H_k_H[k])
        A_pk[k] = np.kron(np.eye(Q), H_k[k] @ G_pk_H[k] @ U_pk[k] @ G_pk[k] @ H_k_H[k])
        a_ck[k] = vectorization(H_k[k] @ G_ck_H[k] @ U_ck_H[k])
        a_pk[k] = vectorization(H_k[k] @ G_pk_H[k] @ U_pk_H[k])
        Phi_ck[k] = 1 * np.trace(U_ck[k] @ G_ck[k] @ G_ck_H[k]) + np.trace(U_ck[k]) - np.log2(np.linalg.det(U_ck[k]))
        Phi_pk[k] = 1 * np.trace(U_pk[k] @ G_pk[k] @ G_pk_H[k]) + np.trace(U_pk[k]) - np.log2(np.linalg.det(U_pk[k]))

    A_prime_ck_tmp.append(A_prime_ck)
    A_ck_tmp.append(A_ck)
    A_pk_tmp.append(A_pk)
    a_ck_tmp.append(a_ck)
    a_pk_tmp.append(a_pk)
    Phi_ck_tmp.append(Phi_ck)
    Phi_pk_tmp.append(Phi_pk)

    A_hat_prime_ck = np.mean(A_prime_ck_tmp,0)
    A_hat_ck = np.mean(A_ck_tmp,0)
    A_hat_pk = np.mean(A_pk_tmp,0)
    a_hat_ck = np.mean(a_ck_tmp,0)
    a_hat_pk = np.mean(a_pk_tmp,0)
    Phi_hat_ck = np.mean(Phi_ck_tmp,0)
    Phi_hat_pk = np.mean(Phi_pk_tmp,0)

    ### Obtain precoder P through CVX tool ###
    mu = np.ones(K)
    X_hat = cp.Variable(K)
    P = [cp.Variable((M, Q), complex=True) for _ in range(K+1)]
    P_vector = cp.vstack([cp.vec(Pk) for Pk in P])
    P_total = cp.hstack(P)

    # Define objective function
    objective_expr = 0
    for k in range(1, K+1):
        A_hat_pk_symmetric = 0.5 * (A_hat_pk[k-1] + A_hat_pk[k-1].conj().T)
        term1 = X_hat[k-1]
        term2 = cp.quad_form(P_vector[k], A_hat_pk_symmetric)
        term3 = 0
        for i in range(1, K+1):
            if i != k:
                term3 += cp.quad_form(P_vector[i], A_hat_pk_symmetric)
        term4 = -2 * cp.real(a_hat_pk[k-1].conj().T @ P_vector[k])
        term5 = Phi_hat_pk[k-1]

        objective_expr += mu[k-1] * (term1 + term2 + term3 + term4 + term5)

    objective = cp.Minimize(objective_expr)

    # 제약 조건 정의
    constraints = []

    # First constraint
    for k in range(1, K+1):
        A_hat_prime_ck_symmetric = 0.5 * (A_hat_prime_ck[k - 1] + A_hat_prime_ck[k - 1].conj().T)
        A_hat_ck_symmetric = 0.5 * (A_hat_ck[k - 1] + A_hat_ck[k - 1].conj().T)
        lhs = cp.sum(X_hat, axis=0) + Q
        rhs1 = cp.quad_form(P_vector[0], A_hat_prime_ck_symmetric)
        rhs2 = 0
        for i in range(1, K+1):
            rhs2 += cp.quad_form(P_vector[i], A_hat_ck_symmetric)
        rhs3 = - 2 * cp.real(a_hat_ck[k-1].conj().T @ P_vector[0]) + Phi_ck[k-1]
        rhs = rhs1 + rhs2 + rhs3
        constraints.append(lhs >= rhs)

    # Second constraint
    constraints.append(cp.norm(P_total, 'fro')**2 <= P_t)

    # Third constraint
    for k in range(K):
        constraints.append(X_hat[k] <= 0)

    # 문제 정의
    prob = cp.Problem(objective, constraints)

    # 문제 풀기
    result = prob.solve()

    # 각 단계에서 P의 값을 저장
    P_curr = [Pk.value for Pk in P]

    # 이전 단계의 P와 현재 단계의 P 간의 Frobenius 노름 제곱을 계산
    error_norm = sum([np.linalg.norm(P_curr[i] - P_prev[i], 'fro')**2 for i in range(K + 1)])
    error_norms.append(error_norm)

    # 현재 단계의 P를 이전 단계의 P로 업데이트
    P_prev = P_curr

# 그래프 그리기
plt.plot(range(iter_num), error_norms)
plt.xlim([0, 10])
plt.xlabel('Iteration')
plt.ylabel('Error Norm Square')
plt.title('Error Norm Square between Iterations')
plt.grid(True)
plt.show()
