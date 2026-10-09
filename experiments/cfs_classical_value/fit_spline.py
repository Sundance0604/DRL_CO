from .fit_models import fit

def fit_spline(rows,knots=6,penalty=10,boundary="natural",terminal="raw"):
    return fit(rows,"spline",penalty,knots=knots,boundary=boundary,terminal=terminal)
