import numpy as np
import tabulate

def rotation(A, b):  # метод вращений
    q = np.column_stack([A, b])
    for i in range(q.shape[0] - 1):
        for j in range(i + 1, q.shape[0]):
            c = q[i, i] / (q[i, i] ** 2 + q[j, i] ** 2) ** (1/2)
            s = q[j, i] / (q[i, i] ** 2 + q[j, i] ** 2) ** (1/2)
            tmp = q[i, :] * c + q[j, :] * s
            q[j, :] = q[i, :] * -s + q[j, :] * c
            q[i, :] = tmp
    x = np.linalg.solve(q[:, :-1], q[:, -1])
    return x
def solution(A):
    x = np.random.uniform(0, 100, size=A.shape[0])
    b = np.dot(A, x)
    _x_r = rotation(A, b)
    print("    ||x - _x_r|| =", np.linalg.norm(x - _x_r))

def main():
    A5 = np.array([[1, 1 / 2, 1 / 3, 1/4,1/5],
                  [1 / 2, 1 / 3, 1 / 4, 1/5,1/6],
                  [1 / 3, 1 / 4, 1 / 5, 1/6,1/7],
                  [1/4,1/5,1/6,1/7,1/8],
                  [1/5,1/6,1/7,1/8,1/9]])
    A3 = np.array([[1, 1 / 2, 1 / 3, ],
                  [1 / 2, 1 / 3, 1 / 4],
                  [1 / 3, 1 / 4, 1 / 5]])
    A4 = np.array([[1, 1 / 2, 1 / 3, 1/4],
                  [1 / 2, 1 / 3, 1 / 4, 1/5],
                  [1 / 3, 1 / 4, 1 / 5, 1/6],
                  [1/4,1/5,1/6,1/7]])
    print('Для матрицы гильберта 5x5:')
    solution(A5)
    print()
    print('Для матрицы гильберта 4x4:')
    solution(A4)
    print('Для матрицы гильберта 3x3:')
    solution(A3)

if __name__ == '__main__':
    main()