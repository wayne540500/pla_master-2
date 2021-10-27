import random
import math
import numpy as np
import matplotlib.pyplot as plt


train_file = open(r'train.txt')
test_file = open(r'test.txt')
text = []

output = open("PLA.out","w")

x1 = []
x2 = []
target = []
draw_x1 = []
draw_x2 = []
draw_y1 = []
draw_y2 = []
test_x1 = []
test_x2 = []
test_y1 = []
test_y2 = []


def sign(n) :
	if(n > 0) :
		return 1
	else :
		return -1

# read file
for i in train_file :
	text.append(i)
for j in text :
	j = j.strip('\n').split(",")
	tmp = 0
	for n in j :
		tmp += 1
		if tmp == 1 :
			x1.append(float(n))
		elif tmp == 2 :
			x2.append(float(n))
		else :
			target.append(float(n))


# PLA train
w = np.array([random.uniform(-10,10),random.uniform(-10,10),random.uniform(-10,10)])
x = np.ndarray(shape = (3,1), dtype = float)
print("Initial w =",w,file = output)

learning_rate = random.random()
new = 0
epoch = 0
epoch_max = 100
stop = 0
while (stop == 0 or epoch_max > epoch) :
	for i in range(200) :
		for j in range(3) :
			if j == 0:
				x[j] = 1
			elif j == 1 :
				x[j] = x1[i]
			else :
				x[j] = x2[i]
		# print(x,file = output)

		if target[i] > 0 :
			draw_x1.append(x1[i])
			draw_y1.append(x2[i])
		else :
			draw_x2.append(x1[i])
			draw_y2.append(x2[i])

		new = sign(w.dot(x))
		if new != target[i] :
			w = w + learning_rate * target[i] * x.T
		else :
			stop = 1
	epoch = epoch + 1

print("Learning rate :",learning_rate,file = output)
print("Epochs :",epoch,file = output)
print("Last w = :",w,file = output)

x1.clear()
x2.clear()
text.clear()

# read file
for i in test_file :
	text.append(i)
for j in text :
	j = j.strip('\n').split(",")
	# print(j)
	tmp = 0
	for n in j :
		tmp += 1
		if tmp == 1 :
			x1.append(float(n))
		elif tmp == 2 :
			x2.append(float(n))


# PLA test
for i in range(10) :
	for j in range(3) :
		if j == 0 :
			x[j] = 1
		elif j == 1 :
			x[j] = x1[i]
		else :
			x[j] = x2[i]
		# print(x)
	ans = sign(w.dot(x))
	if ans > 0 :
		test_x1.append(x1[i])
		test_y1.append(x2[i])
	else :
		test_x2.append(x1[i])
		test_y2.append(x2[i])

	print('test',i+1,":",ans,file = output)

a = np.arange(-20, 20, 0.1)
b = -(a * w.T[1] + w.T[0])/w.T[2]

# draw picture
plt.title("Perceptron Learning Algorithm")
# plt.plot([-20,20],[-20,20])
plt.plot(a,b)
plt.plot(draw_x1, draw_y1, "bo")
plt.plot(draw_x2, draw_y2, "ro")
plt.plot(test_x1, test_y1, "m^")
plt.plot(test_x2, test_y2, "c^")
plt.show()
