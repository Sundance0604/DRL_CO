from .fit_models import fit

def fit_chebyshev(rows,degree=3,penalty=1,terminal="raw"):
    return fit(rows,"chebyshev",penalty,degree=degree,terminal=terminal)
