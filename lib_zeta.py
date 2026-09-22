def zeta_kf(x,n,z_pr):
    return (1-z_pr)+z_pr*2*n*x**2/(n+1)/(x**2+1)

def zeta(x,n,z_pr):
    return 0.99*0.99*zeta_kf(x,n,z_pr)