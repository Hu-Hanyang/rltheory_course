import numpy as np

# Create two 2D numpy arrays
array1 = np.array([[1, 2], [3, 4]])
array2 = np.array([[5, 6], [7, 8]])

# Calculate the inner product
inner_product = np.dot(array1, array2)

# Alternatively, you can use the @ operator
# inner_product = array1 @ array2

print(inner_product)