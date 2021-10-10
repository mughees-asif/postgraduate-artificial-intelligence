start = 1
end = 6

# Q1(a): Print a small multiplication square
for row in range(start, end):
    for col in range(start, end):
        res = row * col
        print(res, end="\t")
    print()

# Q1(b): Print a small multiplication square, even numbers floored to 0
for row in range(start, end):
    for col in range(start, end):
        res = row * col
        if res % 2 == 0:
            res = 0
        print(res, end="\t")
    print()
