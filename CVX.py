import cvxpy as cp
import numpy as np

# 문제 정의를 위한 파라미터 (예시로 랜덤하게 생성합니다)
K = 3  # 사용자 수
n = 4  # precoder 벡터 크기
Q = np.eye(n)  # 예시로 단위 행렬 사용
Pt = 10  # 최대 전력

mu = np.random.rand(K)  # 각 사용자에 대한 가중치
A_p = [np.random.rand(n, n) for _ in range(K)]
A_p_hat = np.random.rand(n, n)
a_p_hat = np.random.rand(n, 1)
Phi_p = np.random.rand(n, n)

A_c_hat = np.random.rand(n, n)
a_c_hat = np.random.rand(n, 1)
Phi_c = np.random.rand(n, n)

X_hat = cp.Variable((K, n, n), PSD=True)
P = [cp.Variable((n, 1)) for _ in range(K)]
print(P)
P_total = cp.hstack(P)

# 목적 함수 정의
objective_expr = 0
for k in range(K):
    term1 = cp.trace(X_hat[k])
    term2 = cp.quad_form(P[k], A_p[k])
    term3 = 0
    for i in range(K):
        if i != k:
            term3 += cp.quad_form(P[i], A_p[k])
    term4 = -2 * cp.real(a_p_hat.T @ P[k])
    term5 = cp.trace(Phi_p)

    objective_expr += mu[k] * (term1 + term2 + term3 + term4 + term5)

objective = cp.Minimize(objective_expr)

# 제약 조건 정의
constraints = []

for k in range(K):
    lhs = cp.sum(X_hat, axis=0) + Q
    rhs = cp.quad_form(P_total, A_c_hat) - 2 * cp.real(a_c_hat.T @ P_total) + cp.trace(Phi_c)
    constraints.append(lhs >= rhs)

constraints.append(cp.trace(P_total @ P_total.T) <= Pt)

for k in range(K):
    constraints.append(X_hat[k] <= 0)

# 문제 정의
prob = cp.Problem(objective, constraints)

# 문제 풀기
result = prob.solve()

# 결과 출력
print("최적 값:", result)
for k in range(K):
    print(f"최적 X_hat[{k}]:\n", X_hat[k].value)
    print(f"최적 P[{k}]:\n", P[k].value)
