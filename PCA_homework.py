import numpy as np

np.set_printoptions(precision=4, floatmode='fixed', suppress=True)

def covariance_analysis(data, sample=False):
    """
    sample=False : 모집단 공분산
    sample=True  : 표본 공분산
    """

    n = data.shape[0]

    # 평균 계산
    mean = np.mean(data, axis=0)

    # 평균 제거
    X = data - mean

    # 모집단/표본 구분
    if sample:
        divisor = n - 1
        ddof = 1
        title = "표본 공분산"
    else:
        divisor = n
        ddof = 0
        title = "모집단 공분산"

    # 수식 이용 계산
    C = (X.T @ X) / divisor

    # NumPy 결과
    C_numpy = np.cov(data, rowvar=False, ddof=ddof)

    print("\n" + "="*50)
    print(title)
    print("="*50)

    print("\n원본 데이터")
    print(data)

    print("\n각 열의 평균")
    print(mean)

    print("\n평균을 뺀 데이터 X")
    print(X)

    print(f"\n{title} 행렬")
    print(C)

    print(f"\n분산(x1) = {C[0,0]:.4f}")
    print(f"분산(x2) = {C[1,1]:.4f}")
    print(f"공분산(x1,x2) = {C[0,1]:.4f}")

    print("\nNumPy 결과와 일치 :", np.allclose(C, C_numpy))


# -------------------------------
# 실습1 : 모집단 공분산
# -------------------------------
data1 = np.array([
    [1.4,0.2],
    [1.4,0.2],
    [1.3,0.2],
    [1.5,0.2],
    [1.4,0.2],
    [1.7,0.4],
    [1.4,0.3],
    [1.5,0.2],
    [1.4,0.2],
    [1.5,0.1]
], dtype=float)

covariance_analysis(data1, sample=False)


# -------------------------------
# 실습2 : 표본 공분산
# -------------------------------
data2 = np.array([
    [1,2],
    [2,6],
    [3,9],
    [5,12]
], dtype=float)

covariance_analysis(data2, sample=True)
