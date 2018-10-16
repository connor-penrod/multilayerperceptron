summation = 0
for x in range(4):
    some_constant = x
    for y in range(4):
        summation += y * some_constant
print(summation)

summation = 0
for x in range(4):
    some_constant = x
    for y in range(4):
        summation += y
    summation *= some_constant
print(summation)