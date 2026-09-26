import numpy as np

GAMMA = 1.4

def primitive_to_conserved(rho, u, p):
    E = p / (GAMMA - 1.0) + 0.5 * rho * u**2
    return np.array([rho, rho*u, E])

def conserved_to_primitive(U):
    rho = np.maximum(U[0], 1e-8)
    u = U[1] / rho
    p = np.maximum((GAMMA - 1.0) * (U[2] - 0.5*rho*u*u), 1e-8)
    return rho, u, p

def flux(U):
    rho, u, p = conserved_to_primitive(U)
    return np.array([rho*u, rho*u*u+p, u*(U[2]+p)])

def rusanov_flux(UL, UR):
    FL, FR = flux(UL), flux(UR)
    rhoL, uL, pL = conserved_to_primitive(UL)
    rhoR, uR, pR = conserved_to_primitive(UR)
    aL = np.sqrt(GAMMA*pL/rhoL)
    aR = np.sqrt(GAMMA*pR/rhoR)
    smax = max(abs(uL)+aL, abs(uR)+aR)
    return 0.5*(FL+FR) - 0.5*smax*(UR-UL)

def solve_riemann(left, right, nx=256, final_time=0.15, cfl=0.45):
    """Finite-volume Rusanov solver for a 1D Euler Riemann problem.

    left/right = (rho, u, p)
    Returns x, rho, u, p at final_time.
    """
    x = np.linspace(-0.5, 0.5, nx)
    dx = x[1]-x[0]
    U = np.zeros((3, nx))
    mid = nx//2
    U[:, :mid] = primitive_to_conserved(*left)[:, None]
    U[:, mid:] = primitive_to_conserved(*right)[:, None]

    t = 0.0
    while t < final_time:
        rho, u, p = conserved_to_primitive(U)
        a = np.sqrt(GAMMA*p/rho)
        dt = cfl*dx/np.max(np.abs(u)+a)
        dt = min(dt, final_time-t)

        # transmissive ghost cells
        Ug = np.pad(U, ((0,0),(1,1)), mode="edge")
        F = np.zeros((3, nx+1))
        for i in range(nx+1):
            F[:, i] = rusanov_flux(Ug[:, i], Ug[:, i+1])

        U = U - dt/dx*(F[:,1:] - F[:,:-1])
        t += dt

        # safety floor
        rho, u, p = conserved_to_primitive(U)
        U[0] = np.maximum(rho, 1e-8)
        U[2] = p/(GAMMA-1.0) + 0.5*U[0]*u*u

    rho, u, p = conserved_to_primitive(U)
    return x, rho, u, p
