import os
import numpy as np
from xml.etree import ElementTree as et
from flask import Flask, render_template
from flask import jsonify
from flask import request

app = Flask(__name__, template_folder='.')

@app.route('/', methods=['GET'])
def index():
  return render_template('index.html')

@app.route('/ahp', methods=['POST'])
def calcular_ahp():

    comparisons = ([])
    comparisons.append(float(request.json['comparison1to2']))
    comparisons.append(float(request.json['comparison1to3']))
    comparisons.append(float(request.json['comparison2to3']))

    js = calc(comparisons)

    return js

def calc(comparisons):
    num_comparisons = len(comparisons)
    matrix = get_matrix(comparisons, num_comparisons)
    total = get_total_criterio(matrix, num_comparisons)
    normalization = get_normalization(matrix, total, num_comparisons)
    eigens = get_eigens(normalization, num_comparisons)
    eigen = get_eigen(eigens, total, num_comparisons)
    ci = get_ci(eigen, num_comparisons)
    cr = get_rc(ci, num_comparisons)

    js = jsonify(
        eigens=str(eigens),
        eigen=eigen,
        ci=ci,
        cr=cr
    )
    print(js)
    return js


# Receives the vector with the comparisons weights and the number of comparisons
# Returns the matrix with the weight of the comparisons with the counterweights
def get_matrix(comparisons, num_comparisons):
    d = num_comparisons * num_comparisons
    matrix = np.zeros(d, dtype=float).reshape(num_comparisons, num_comparisons)

    k = 0
    for i in range(0, num_comparisons):
        for j in range(0, num_comparisons):
            # diagonal
            if i == j:
                matrix[i][j] = 1.0
            elif i < j:
                matrix[i][j] = comparisons[k]
                k = k + 1
            elif i > j:
                matrix[i][j] = 1.0 / matrix[j][i]

    return matrix

# Receives matrix with the weights of the comparisons and the number of comparisons
# Returns the total by comparisons
def get_total_criterio(matrix, num_comparisons):
    total = np.zeros(num_comparisons, dtype=float)
    for i in range(0, num_comparisons):
        total[i] = 0
        for j in range(0, num_comparisons):
            total[i] += matrix[j][i]

    return total

# Receives the matrix with the weights of the comparisons, the vector with the total per comparisons and the number of comparisons
# Returns the normalized array
def get_normalization(matrix, total, num_comparisons):
    d = num_comparisons * num_comparisons
    normalizacao = np.zeros(d, dtype=float).reshape(num_comparisons, num_comparisons)
    for i in range(0, num_comparisons):
        for j in range(0, num_comparisons):
            normalizacao[j][i] = matrix[j][i] / total[i]

    return normalizacao

# Receives the normalized matrix and the number of comparisons
# Returns the vector with the values of eigen
def get_eigens(normalizacao, num_comparisons):
    eigens = np.zeros(num_comparisons, dtype=float)
    for i in range(0, num_comparisons):
        eigens[i] = 0
        for j in range(0, num_comparisons):
            eigens[i] = eigens[i] + normalizacao[i][j]
        eigens[i] = eigens[i] / num_comparisons

    return eigens

# Receives the vector of eigen, the vector with the total per criterion and the number of comparisons
# Returns the main number of eigen
def get_eigen(eigens, total, num_comparisons):
    eigen = 0
    for i in range(0, num_comparisons):
        eigen = eigen + (eigens[i] * total[i])

    return eigen

# Receives the main number of eigen and the number of comparisons
# Returns the consistency index
def get_ci(eigen, num_comparisons):
    return (eigen - num_comparisons) / (num_comparisons - 1)

# Receives the consistency index and the number of comparisons
# Returns the consistency rate
def get_rc(ci, num_comparisons):
    ri = np.array([0.0, 0.0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49])

    return ci / ri[num_comparisons - 1]

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)