from .fit_models import fit

def fit_linear(rows,penalty=1):
    return fit(rows,"linear",penalty)
