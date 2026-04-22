import torch


def checkpoint_sequential(
    functions,
    input,
):
    def wrap_tuple(a):
        return (a,) if type(a) is not tuple else a

    def exec(func, a):
        return wrap_tuple(func(*a))

    def get_wrap_exec(func):
        def wrap_exec(*a):
            return exec(func, a)

        return wrap_exec

    input = wrap_tuple(input)

    is_grad_enabled = torch.is_grad_enabled()

    if is_grad_enabled:
        for func in functions:
            input = torch.utils.checkpoint.checkpoint(get_wrap_exec(func), *input)
    else:
        for func in functions:
            input = exec(func, input)
    return 

