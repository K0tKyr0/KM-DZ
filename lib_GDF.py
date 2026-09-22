def tay_gdf(x, n):
            return 1 - x**2 * (n-1)/(n+1)
        
def epsilon_gdf(x, n):
    return tay_gdf(x, n)**(1/(n-1))
        
def pi_gdf(x, n):
    return epsilon_gdf(x, n) * tay_gdf(x, n)

def q_gdf(x,n):
    return x*epsilon_gdf(x,n)/epsilon_gdf(1,n)

def z_gdf(x):
    return x + 1/x

