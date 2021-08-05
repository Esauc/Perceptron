import numpy as np

class Perceptron:

  def __init__(self, input_values, output_values, learning_rate, activation_function):
    self.input_values = input_values
    self.output_values = output_values
    self.learning_rate = learning_rate
    self.activation_function = activation_function
    self.W = np.random.rand(input_values.shape[1]) #Pesos W gerados de forma aleatória
    self.theta = np.random.rand(1)[0] #gerado de forma aleatória

  def train(self):
    
    initialW = [self.W[0], self.W[1], self.W[2]]

    initialTheta = self.theta

    epochs = 1
    error = True
    print(f'Initial W: {self.W}')
    print(f'Initial Theta: {self.theta}')
    print(f'[EPOCH {epochs}]')

    while error:
      error = False
      for x, d in zip(self.input_values, self.output_values):

        u = np.dot(x, self.W) - self.theta
        y = self.activation_function.g(u)

        print(f'Input: {x}, Output: {y}, Expected: {d}')

        if y != d:
          print(f'Output is different from expected, recalculating W!')
          print(f'Actual W: {self.W}')
          print(f'Actual Theta: {self.theta}')

          self.W = self.W + self.learning_rate * (d - y) * x
          self.theta = self.theta + self.learning_rate * (d - y) * -1
          error = True

          print(f'New W: {self.W}')
          print(f'New Theta: {self.theta}')
          print('')

          break

      epochs += 1

      print('')
    print('DONE TRAINING')
    print(f'Initial W: {initialW}')
    print(f'Initial Theta: {initialTheta}')

  def evaluate(self, input_value):
    u = np.dot(input_value, self.W) - self.theta
    return self.activation_function.g(u)
