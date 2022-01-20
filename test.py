from re import A
import torch

width = 8
height = 8

high = torch.eye(8)

visited = torch.zeros_like(high)
final = torch.zeros_like(high).float()

high[3, 3] = 0.0
high[0, 3] = 1.0
high[1, 4] = 1.0
high[2, 5] = 1.0
high[2, 6] = 1.0
high[3, 7] = 1.0
high[0, 6] = 1.0

low = high.clone()

low[3, 3] = 1.0
low[1, 6] = 1.0
low[5, 1] = 1.0
low[6, 2] = 1.0

def connected(x, y):
    left = x - 1
    top = y - 1

    if left < 0 or top < 0:
        return False
    
    return final[left, top] > 0 or final[left, y] > 0 or final[x, top] > 0

def trace(x:int, y:int):
    right = x + 1
    bottom = y + 1
    if right >= width or bottom >= height:
        return

    pass_high = high[x, y] > 0.0
    pass_low = low[x, y] > 0.0

    if pass_high:
        final[x, y] = high[x, y]
    elif pass_low and not pass_high:
        if connected(x, y):
            print(x, y)
            final[x, y] = low[x, y]
    
    trace(right, y)
    trace(x, bottom)    
    trace(right, bottom)

trace(0, 0)

print(low)
print(high)
print(final)