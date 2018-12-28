import torch

def calculate_pad(hei, wid, hei_pad, wid_pad):
    left = (wid_pad - wid) // 2
    right = wid_pad - wid - left
    top = (hei_pad - hei) // 2
    bottom = hei_pad - hei - top
    return [left, right, top, bottom]

# concat 3D/4D variable
# dim = 0: 3D; = 1: 4D
def concat_with_pad(seq, dim):
    # get maximum size
    hei_pad = 0
    wid_pad = 0
    for input in seq:
        hei_pad = max(hei_pad, input.size(dim + 1))
        wid_pad = max(hei_pad, input.size(dim + 2))

    # pad each input
    output = []
    for input in seq:
        pad = calculate_pad(input.size(dim + 1), input.size(dim + 2), hei_pad, wid_pad)
        input_pad = torch.nn.functional.pad(input, pad)
        output.append(input_pad)
    return torch.cat(output, dim)