"""
    An implementation of Strassen algorithm with recursion-style in Python version (only NumPy)
    Data: 2023-08-29
    Author: Wenxuan Zeng
    Reference:
    * https://zhuanlan.zhihu.com/p/268392799
    * https://en.wikipedia.org/wiki/Strassen_algorithm
    * https://blog.csdn.net/weixin_40982849/article/details/121338011
"""

import numpy as np


# divide the whole matrix into 4 parts uniformly
def divide(mat, x_idx, y_idx):
    len_divide = mat.shape[0] // 2
    mat_divide = np.zeros((len_divide, len_divide))
    for i in range(len_divide):
        for j in range(len_divide):
            mat_divide[i][j] = mat[i + x_idx * len_divide][j + y_idx * len_divide]

    return mat_divide


# merge 4 submatrices into one matrix
def matrix_merge(mat1, mat2, mat3, mat4):
    len_divide = mat1.shape[0]
    len_merge = mat1.shape[0] * 2
    mat_merge = np.zeros((len_merge, len_merge))
    merge_way = 2  # choose one implementation style to merge the matrices
    if merge_way == 1:
        # way1: loop over the large matrix
        for i in range(len_merge):
            for j in range(len_merge):
                if i < len_divide and j < len_divide:
                    mat_merge[i][j] = mat1[i][j]
                if i < len_divide and j >= len_divide:
                    mat_merge[i][j] = mat2[i][j - len_divide]
                if i >= len_divide and j < len_divide:
                    mat_merge[i][j] = mat3[i - len_divide][j]
                if i >= len_divide and j >= len_divide:
                    mat_merge[i][j] = mat4[i - len_divide][j - len_divide]

    elif merge_way == 2:
        # way2: loop over each submatrices
        for i in range(len_divide):
            for j in range(len_divide):
                mat_merge[i][j] = mat1[i][j]
                mat_merge[i][j + len_divide] = mat2[i][j]
                mat_merge[i + len_divide][j] = mat3[i][j]
                mat_merge[i + len_divide][j + len_divide] = mat4[i][j]

    return mat_merge


# the core implementation of Strassen algorithm
def strassen(a, b):
    # recursion exit: the matrix cannot be further split, perform multiplication
    if a.shape[0] == 1:
        return a * b
    else:  # recursion process
        # divide matrices into 4 parts
        a00, a01, a10, a11 = divide(a, 0, 0), divide(a, 0, 1), divide(a, 1, 0), divide(a, 1, 1)
        b00, b01, b10, b11 = divide(b, 0, 0), divide(b, 0, 1), divide(b, 1, 0), divide(b, 1, 1)

        # compute 10 immediate matrices
        s1 = b01 - b11
        s2 = a00 + a01
        s3 = a10 + a11
        s4 = b10 - b00
        s5 = a00 + a11
        s6 = b00 + b11
        s7 = a01 - a11
        s8 = b10 + b11
        s9 = a00 - a10
        s10 = b00 + b01

        # compute 7 matrix multiplications
        # note! Using strassen function to achieve recursion algorithm instead of np.matmul!!!
        p1 = strassen(a00, s1)
        p2 = strassen(s2, b11)
        p3 = strassen(s3, b00)
        p4 = strassen(a11, s4)
        p5 = strassen(s5, s6)
        p6 = strassen(s7, s8)
        p7 = strassen(s9, s10)

        # compute 4 adds to obtain the final 4 parts
        c00 = p5 + p4 - p2 + p6
        c01 = p1 + p2
        c10 = p3 + p4
        c11 = p5 + p1 - p3 - p7

        # merge 4 submatrices into one matrix
        mat_merge = matrix_merge(c00, c01, c10, c11)

    return mat_merge


def main():
    # define the input matrices a and b
    a = np.array([[1, 2, 4, 2],
                  [3, 4, 2, 1],
                  [4, 4, 7, 2],
                  [2, 5, 1, 4]])
    b = np.array([[3, 5, 1, 2],
                  [2, 7, 4, 3],
                  [6, 3, 4, 2],
                  [8, 3, 2, 5]])
    res_actual = np.matmul(a, b)
    print('Actual product result:\n', res_actual)

    res_strassen = strassen(a, b)
    print('Strassen product result:\n', res_strassen)

    if np.array_equal(res_actual, res_strassen):
        print('The result of Strassen is correct!')
    else:
        print('The result of Strassen is wrong!')


if __name__ == '__main__':
    main()
