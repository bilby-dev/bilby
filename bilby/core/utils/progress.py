def get_progress_bar(module='tqdm'):
    """
    TODO: Write proper docstring
    """
    if module in ['tqdm']:
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(x, *args, **kwargs):
                return x
        return tqdm
    elif module in ['tqdm_notebook']:
        try:
            from tqdm import tqdm_notebook as tqdm
        except ImportError:
            def tqdm(x, *args, **kwargs):
                return x
        return tqdm
