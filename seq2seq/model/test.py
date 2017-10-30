# losses = []
# import matplotlib.pyplot as pyplot
# #%pyplot
# import matplotlib.pyplot as plt
# def step():
#     return 6
#
# for epoch in range(5):
#     loss = step()
#     print("At Epoch : {:5},Get loss : {:10}\n".format(epoch, loss))
#     losses.append(loss)
#     plt.plot(epoch, loss,"*-")
#     plt.show()

from pylab import *
import random
import time

fig, ax = plt.subplots()

grid(True)
plt.ion()
x = []
y = []
i = 0
while True:
    if True:
        dat = 10 * random.random()
        x.append(i)
        y.append(dat)
        ax.plot(x, y, 'b')
        plt.pause(0.0001)


    i += 1
    time.sleep(0.01)

print('over')