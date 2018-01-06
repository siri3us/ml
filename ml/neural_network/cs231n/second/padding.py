#-*- coding: utf-8 -*-

def find_1d_padding(array_size, filter_size, stride):
    """
    This function finds padding for 1D arrays.
    
    Inputs:
    - array_size:   size of the provided array
    - filter_size:  size of filter
    - stride:       stride used for convolution (cross correlation)
    
    Returns a tuple of:
    - padding (left_pad, right_pad): tuple of ints
        - left_pad:  left padding for the array
        - right_pad: right padding for the array
    - output_size: int; the output size of convolution in case of the found padding equals
            (array_size + left_pad + right_pad - filter_size) / stride + 1
    """
    assert array_size > 0
    assert filter_size > 0
    assert stride > 0
    
    # It is possible that array_size < filter_size. In that case array first needs to be padded
    init_left_pad = init_right_pad = 0
    if filter_size > array_size:
        dif = filter_size - array_size
        init_left_pad  = (dif + 1) // 2
        init_right_pad = dif - init_left_pad
        assert init_left_pad >= init_right_pad
        array_size += init_left_pad + init_right_pad

    for left_pad in range(filter_size):
        for right_pad in range(left_pad + 1):
            if check_padding(array_size, filter_size, stride, (left_pad, right_pad)):
                output_size = 1 + (array_size + left_pad + right_pad - filter_size) // stride
                return (init_left_pad + left_pad, init_right_pad + right_pad), output_size
    assert False, 'Appropriate padding not found'


def find_2d_padding(image_size, filter_size, stride):
    """
    This function finds padding for 2D arrays. For that it finds padding along height and width.
    
    Inputs:
    - image_size:   tuple (H, W) of image height and width
    - filter_size:  tuple (HH, WW) of filter height and width
    - stride:       int, equals for both height and width
    
    Returns a tuple of:
    - (H_pad, W_pad): tuple of ints
        - H_pad is a tuple (upp_pad, low_pad) for height paddding
        - W_pad is a tuple (left_pad, right_pad) for width padding
    - (H_output, W_output): tuple of ints
        - H_output is the output height of convolution
        - W_output is the output width of convolution
    """
    H,  W  = image_size
    HH, WW = filter_size
    H_pad, H_output = find_1d_padding(H, HH, stride)
    W_pad, W_output = find_1d_padding(W, WW, stride)
    return (H_pad, W_pad), (H_output, W_output)

def parse_padding(padding):
    """Accepts a sequence of padding values and turns them into standard format"""
    if isinstance(padding, int):
        padding = [padding]
    elif isinstance(padding, tuple):
        assert len(padding) == 2
        padding = [padding]
    assert isinstance(padding, list), 'Padding must be provided in a list.'
    parsed = []
    for value in padding:
        if isinstance(value, int):
            parsed.append((value, value))
        elif isinstance(value, (tuple)):
            assert len(value) == 2
            parsed.append(value)
        else:
            assert False, 'Unknown padding type "{}"'.format(type(value).__name__)
    return parsed
        
def check_padding(array_size, filter_size, stride, pad):
    """
    This function checks that given array padding is correct.
    
    Inputs:
    - array_size: int; array size
    - filter_size: int; size of filter
    - stride: int; stride of convolution (cross-correlation)
    - pad: int or tuple of 2 ints; int the first case left and right padding are the same, in the second case
        they equal the values in the tuple respectively.
        
    Returns True if padding is correct. Otherwise returns False.
    """
    pad = parse_padding(pad)
    assert len(pad) == 1
    left_pad, right_pad = pad[0]
    if (array_size + left_pad + right_pad - filter_size) % stride == 0:
        return True
    return False
    
    
    
"""print(parse_padding(1))
print(parse_padding((1, 2)))
print(parse_padding([1, 2, 3]))
print(parse_padding([1, (2, 1), 1]))
check_padding(4, 2, 2, (1, 2))

assert find_1d_padding(3, 4, 1) == ((1, 0), 1)
assert find_1d_padding(5, 2, 3) == ((0, 0), 2)"""
