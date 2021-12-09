import numpy as np

def get_shape_str(input):
    shape_str = ''
    N = input.ndim
    for i in range(N):
        shape_str = shape_str + '{}x'.format(input.shape[i])
    shape_str = shape_str[:-1]
    return shape_str

def print_values(input):
	values_str = 'Min: {:.{n}f} Mean: {:.{n}f} Max: {:.{n}f} Std: {:.4f} | Shape: {}'.format( np.min(input), np.mean(input), np.max(input), np.std(input), get_shape_str(input), n=2 )
	#values_str = 'Min: {:.2f} Mean: {:.2f} Max: {:.2f} Std: {:.4f} | Shape: {}'.format( np.min(input), np.mean(input), np.max(input), np.std(input), get_shape_str(input) )
	print(values_str)