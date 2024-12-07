



















###################Tips:
'''
x1 = torch.randint(low=0, high=3, size=(2,3,4,5)) 
x2 = torch.randint(low=0, high=3, size=(2,3,4,5)) 

For two tensors of the same shape, the batches and channels match up one-to-one. This means:

The first batch of x1 pairs with the first batch of x2.
The second batch of x1 pairs with the second batch of x2.
And similarly for channels:

The first channel of x1 pairs with the first channel of x2.
The second channel of x1 pairs with the second channel of x2.
The third channel of x1 pairs with the third channel of x2.
Without any special operations, there's no mixing of batches or channels across different indices.
'''
#example
import torch
x1 = torch.randint(low=0, high=3, size=(2,3,4,5)) 
x2 = torch.randint(low=0, high=3, size=(2,3,4,5))
print(x1.T.shape)
print((x1@x2.transpose(-1,-2)).shape)

print("")
x1 = torch.randint(low=0, high=3, size=(1,1,4,5)) 
x2 = torch.randint(low=0, high=3, size=(2,3,4,5))
print((x1@x2.transpose(-1,-2)).shape)

print("")
x1 = torch.randint(low=0, high=3, size=(1,1,4,5)) 
x2 = torch.randint(low=0, high=3, size=(2,3,4,5))
print((x1@x2.transpose(-1,-2)).shape)
