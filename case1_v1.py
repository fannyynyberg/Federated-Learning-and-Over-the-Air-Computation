import numpy as np
import matplotlib.pyplot as plt

w0 = [1.0, 2.0, 3.0]  #Vikter för klient 1, 2 och 3
gradients = [0.1, 0.2, 0.15] # Gradienter beräknade för varje klient
mu = 0.01  #Inlärningshastighet

def FL_round(w0, gradients, mu):
 
    #Beräkna global medelvikt wG(1)
    wG = np.mean(w0)

    #Uppdatera vikterna på varje klient
    updated_weights = [wG - mu * grad for grad in gradients]

    return updated_weights, wG

#Simulera flera rundor
num_rounds = 100
weights = w0
global_weights = []

for _ in range(num_rounds):
    #Kör en runda FL
    weights, global_weight = FL_round(weights, gradients, mu)
    #Spara det globala medelvärdet
    global_weights.append(global_weight)

#Plotta konvergensen av den globala vikten
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_rounds + 1), global_weights, marker='o')
plt.title('Konvergens av den globala medelvikten')
plt.xlabel('Runda')
plt.ylabel('Global medelvikt')
plt.grid()
plt.show()
