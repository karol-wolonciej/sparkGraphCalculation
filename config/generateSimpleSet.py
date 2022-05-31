from random import randint

n = 10000

m = int(n * 1.1)

points = []

maxNumber = 1000

getNumber = lambda: randint(0, maxNumber)
getPoint = lambda: (getNumber(), getNumber())


for i in range(m):
    points.append(getPoint())

points = list(set(points))
dataset = points[0:n]

print(len(dataset))


f = open("randomSet.txt", "a")

for point in dataset:
    f.write(f'{point[0]} {point[1]}\n')

f.close()