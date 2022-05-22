


M = [[0,   0,   1/2, 0],
     [1/2, 0,   0,   0],
     [0,   1/2, 0,   0],
     [1/2, 1/2, 1/2, 0]]

M_teleport = [[0,    0,   1/2, 1/4],
              [1/2,  0,   0,   1/4],
              [0,    1/2, 0,   1/4],
              [1/2,  1/2, 1/2, 1/4]]


B = [[1/4, 1/4, 1/4, 1/4],
     [1/4, 1/4, 1/4, 1/4],
     [1/4, 1/4, 1/4, 1/4],
     [1/4, 1/4, 1/4, 1/4]]

r = [1/4,
     1/4,
     1/4,
     1/4]

def print_matrix(A):
    for row in A:
        r = ""
        for col in row:
            r += str(round(col,2)) + ", "
        print(r)
    print()

def print_vector(r):
    print(r, sum(r))

def multiply_matrix(M, v):
    return [[col * v for col in row] for row in M]

def multiply_matrix_vector(M, r):
    return [sum([col * r_e for (col, r_e) in zip(row, r)]) for row in M]

def add_matrixes(A, B):
    return [[col_a + col_b for (col_a, col_b) in zip(row_a, row_b)] for (row_a, row_b) in zip(A, B)]

def do_iteration(M, B, r, b):
    MB = add_matrixes(multiply_matrix(M, b),  multiply_matrix(B, 1-b))
    return multiply_matrix_vector(MB, r)



print_matrix(M)

print_matrix(B)

print_vector(r)

print_matrix(multiply_matrix(B, 2))

print_vector(multiply_matrix_vector(M, r))

print_matrix(add_matrixes(M, B))


# b = 1
b = 1
first_r = do_iteration(M, B, r, b)
second_r = do_iteration(M, B, first_r, b)
print_vector(first_r)
print_vector(second_r)
print()

# b = 0.8
b = 0.8
first_r = do_iteration(M, B, r, b)
second_r = do_iteration(M, B, first_r, b)
print_vector(first_r)
print_vector(second_r)
print()

# random teleport
b = 1
first_r = do_iteration(M_teleport, B, r, b)
second_r = do_iteration(M_teleport, B, first_r, b)
print_vector(first_r)
print_vector(second_r)
print()

