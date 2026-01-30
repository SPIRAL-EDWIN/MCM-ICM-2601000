M = 10**8                 #Total cargo volume (tons)
E_E = 179000 * 3          #Elevator annual cargo capacity (tons/year)
P_avg = 1506              #Average payload capacity per rocket (tons/launch)
E_r = 125 * P_avg         #Rocket annual cargo capacity (tons/year)
Myd = 10**19
C_e1 = 2 * 10**6          #Marginal cost of Space Elevators transportation - Part1 (USD/ton)
C_e2 = 1.5 * 10**6        #Marginal cost of Space Elevators transportation - Part2 (USD/ton)
C_E = C_e1 + C_e2         #Marginal cost of Space Elevators transportation (USD/ton)
C_R = 10**9               #Rocket marginal cost (USD/ton)
F_R = 10**8               #Rocket fixed cost (USD/year)
F_E = 2 * 10**7           #Elevator fixed cost (USD/year)
ap = 2                    #alpha: The proportion of Space Elevator Transportation
for a in range(0,10001):
    T1 = a / 10000 * M / E_E
    T2 = (10000 - a) / 10000 * M / E_r
    T3 = max(T1,T2)
    cost = a/10000 * M * C_E + (10000 - a)/10000 * M * C_R + (1-a/10000) * F_R * T3 + a/10000 * F_E * T3
    if 0.6 * cost + 0.4 * T3 <= Myd:
        Myd = 0.6 * cost + 0.4 * T3
        ap = a / 10000
print(str(ap*100)+"%")