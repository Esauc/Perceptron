import pandas as pd

from activation_function import SignFunction
from perceptron import Perceptron


print('------------------------------------')

dataset = pd.read_csv('database/dataset-treinamento.csv')

n = 3 #Número de entradas

# dataset.iloc[LINHA INICIAL (Inclusive) : LINHA FINAL (exclusive) , COLUNA INICIAL (inclusive) : COLUNA FINAL (exclusive)]
#Só : significa todas as linhas/colunas
X = dataset.iloc[:, 0:n].values # ENTRADAS
d = dataset.iloc[:, n:].values  #SAÍDAS

p = Perceptron(X, d, 0.01, SignFunction)
#0.1 é LEARNING RATE
#X valores de entrada
#d valores de saída

#p.train()

print('')
#Treinamento 1
p.W = [2.82009317, 4.84372423, -1.25528586]
p.theta = -5.02914167
print(f'T1 Amostra 1, saida {p.evaluate([-0.3665, 0.0620, 5.9891])}')
print(f'T1 Amostra 2, saida {p.evaluate([-0.7842, 1.1267, 5.5912])}')
print(f'T1 Amostra 3, saida {p.evaluate([0.3012, 0.5611, 5.8234])}')
print(f'T1 Amostra 4, saida {p.evaluate([0.7757, 1.0648, 8.0677])}')
print(f'T1 Amostra 5, saida {p.evaluate([0.1570, 0.8028, 6.3040])}')
print(f'T1 Amostra 6, saida {p.evaluate([-0.7014, 1.0316, 3.6005])}')
print(f'T1 Amostra 7, saida {p.evaluate([0.3748, 0.1536, 6.1537])}')
print(f'T1 Amostra 8, saida {p.evaluate([-0.6920, 0.9404, 4.4058])}')
print(f'T1 Amostra 9, saida {p.evaluate([-1.3970, 0.7141, 4.9263])}')
print(f'T1 Amostra 10, saida {p.evaluate([-1.8842, -0.2805, 1.2548])}')

print('')
#Treinamento 2
p.W = [2.83587799, 4.90556303, -1.2651401]
p.theta = -5.06673522
print(f'T2 Amostra 1, saida {p.evaluate([-0.3665, 0.0620, 5.9891])}')
print(f'T2 Amostra 2, saida {p.evaluate([-0.7842, 1.1267, 5.5912])}')
print(f'T2 Amostra 3, saida {p.evaluate([0.3012, 0.5611, 5.8234])}')
print(f'T2 Amostra 4, saida {p.evaluate([0.7757, 1.0648, 8.0677])}')
print(f'T2 Amostra 5, saida {p.evaluate([0.1570, 0.8028, 6.3040])}')
print(f'T2 Amostra 6, saida {p.evaluate([-0.7014, 1.0316, 3.6005])}')
print(f'T2 Amostra 7, saida {p.evaluate([0.3748, 0.1536, 6.1537])}')
print(f'T2 Amostra 8, saida {p.evaluate([-0.6920, 0.9404, 4.4058])}')
print(f'T2 Amostra 9, saida {p.evaluate([-1.3970, 0.7141, 4.9263])}')
print(f'T2 Amostra 10, saida {p.evaluate([-1.8842, -0.2805, 1.2548])}')

print('')
#Treinamento 3
p.W = [2.87434396, 4.94373118, -1.28505414]
p.theta = -5.15415993
print(f'T3 Amostra 1, saida {p.evaluate([-0.3665, 0.0620, 5.9891])}')
print(f'T3 Amostra 2, saida {p.evaluate([-0.7842, 1.1267, 5.5912])}')
print(f'T3 Amostra 3, saida {p.evaluate([0.3012, 0.5611, 5.8234])}')
print(f'T3 Amostra 4, saida {p.evaluate([0.7757, 1.0648, 8.0677])}')
print(f'T3 Amostra 5, saida {p.evaluate([0.1570, 0.8028, 6.3040])}')
print(f'T3 Amostra 6, saida {p.evaluate([-0.7014, 1.0316, 3.6005])}')
print(f'T3 Amostra 7, saida {p.evaluate([0.3748, 0.1536, 6.1537])}')
print(f'T3 Amostra 8, saida {p.evaluate([-0.6920, 0.9404, 4.4058])}')
print(f'T3 Amostra 9, saida {p.evaluate([-1.3970, 0.7141, 4.9263])}')
print(f'T3 Amostra 10, saida {p.evaluate([-1.8842, -0.2805, 1.2548])}')

print('')
#Treinamento 4
p.W = [2.89698173, 4.9952095, -1.29381719]
p.theta = -5.19851776
print(f'T4 Amostra 1, saida {p.evaluate([-0.3665, 0.0620, 5.9891])}')
print(f'T4 Amostra 2, saida {p.evaluate([-0.7842, 1.1267, 5.5912])}')
print(f'T4 Amostra 3, saida {p.evaluate([0.3012, 0.5611, 5.8234])}')
print(f'T4 Amostra 4, saida {p.evaluate([0.7757, 1.0648, 8.0677])}')
print(f'T4 Amostra 5, saida {p.evaluate([0.1570, 0.8028, 6.3040])}')
print(f'T4 Amostra 6, saida {p.evaluate([-0.7014, 1.0316, 3.6005])}')
print(f'T4 Amostra 7, saida {p.evaluate([0.3748, 0.1536, 6.1537])}')
print(f'T4 Amostra 8, saida {p.evaluate([-0.6920, 0.9404, 4.4058])}')
print(f'T4 Amostra 9, saida {p.evaluate([-1.3970, 0.7141, 4.9263])}')
print(f'T4 Amostra 10, saida {p.evaluate([-1.8842, -0.2805, 1.2548])}')

print('')
#Treinamento 5
p.W = [2.88117214, 4.97760613, -1.2866221]
p.theta = -5.16634907
print(f'T5 Amostra 1, saida {p.evaluate([-0.3665, 0.0620, 5.9891])}')
print(f'T5 Amostra 2, saida {p.evaluate([-0.7842, 1.1267, 5.5912])}')
print(f'T5 Amostra 3, saida {p.evaluate([0.3012, 0.5611, 5.8234])}')
print(f'T5 Amostra 4, saida {p.evaluate([0.7757, 1.0648, 8.0677])}')
print(f'T5 Amostra 5, saida {p.evaluate([0.1570, 0.8028, 6.3040])}')
print(f'T5 Amostra 6, saida {p.evaluate([-0.7014, 1.0316, 3.6005])}')
print(f'T5 Amostra 7, saida {p.evaluate([0.3748, 0.1536, 6.1537])}')
print(f'T5 Amostra 8, saida {p.evaluate([-0.6920, 0.9404, 4.4058])}')
print(f'T5 Amostra 9, saida {p.evaluate([-1.3970, 0.7141, 4.9263])}')
print(f'T5 Amostra 10, saida {p.evaluate([-1.8842, -0.2805, 1.2548])}')

print('')
print('')

#print('TESTES')
#print(f'Input: [1, 1, 1], output {p.evaluate([1, 1, 1])}')
#print(f'Input: [0, 1, 0], output {p.evaluate([0, 1, 0])}')
#print(f'Input: [1, 0], output {p.evaluate([1, 0])}')
#print(f'Input: [0, 0], output {p.evaluate([0, 0])}')