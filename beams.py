

def ss_center_point(x, L, P, E, I):
    
    if x <= L / 2:
        y = P * x * (3 * L**2 - 4 * x**2) / (48 * E * I)
    else:
        z = L - x
        y = P * z * (3 * L**2 - 4 * z**2) / (48 * E * I)

    return y


def ss_off_center_point(x, L, P, E, I, a):
    
    b = L - a

    if x <= a:
        y = P * b * x * (L**2 - b**2 - x**2) / (6 * L * E * I)
    else:
        y = P * a * (L - x) * (L**2 - a**2 - (L - x)**2) / (6 * L * E * I)

    return y


def ss_udl(x, L, w, E, I):
   
    if x <= L / 2:
        y = w * x * (L**3 - 2 * L * x**2 + x**3) / (24 * E * I)
    else:
        z = L - x
        y = w * z * (L**3 - 2 * L * z**2 + z**3) / (24 * E * I)

    return y


def cantilever_point(x, L, P, E, I):
    
    y = P * x**2 * (3 * L - x) / (6 * E * I)
    return y


def cantilever_udl(x, L, w, E, I):
    
    y = w * x**2 * (6 * L**2 - 4 * L * x + x**2) / (24 * E * I)
    return y
