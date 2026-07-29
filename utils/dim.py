

def ensure_batch_first(x, batch_first):
    squeeze = False
    transpose = False
    if x.dim() == 2:
        x = x.unsqueeze(dim=0)
        squeeze = True
    elif x.dim() == 3:
        if not batch_first:
            x = x.transpose(0, 1)
            transpose = True
    else:
        raise ValueError(f'invalid dim of x {x.dim()}')

    return (transpose, squeeze), x


def restore_shape(x, transpose, squeeze):
    if transpose:
        x = x.transpose(0, 1)

    if squeeze:
        x = x.squeeze(0)

    return x
