import numpy as np

def layered_potential_anisotropy_tilde(wx, wy, sxx, sxy, sxz, syy, syz, szz, thicknesses):
    """
    Caλculates phi_tilde for unitary source at the surface.

    wx, wy
    """
    i0 = np.where((wx == 0) & (wy == 0))[0]
    wx[i0] = 1.0
    wy[i0] = 1.0
    N = len(sxx)

    a = szz[:, None]
    b = sxz[:, None]*wx+syz[-1][:, None]*wy
    c = sxx[:, None]*wx*wx+2*sxy[:, None]*wx*wy+syy[:, None]*wy*wy

    aλ = np.sqrt(a*c-b*b)
    tanh = np.tanh(aλ[:-1] * thicknesses[:, None] / a)
    Q = 1/aλ[-1]
    for i in range(N-2, -1, -1):
        aλQ = aλ[i]*Q
        Q = (aλQ+tanh[i])/(aλ*(aλQ*tanh[i]+1))
    Q[i0] = 0.0
    return Q/(2*np.pi)

def grad_layered_potential_anisotropy_tilde(wx, wy, sxx, sxy, sxz, syy, syz, szz, thicknesses):
    """
    Caλculates phi_tilde for unitary source at the surface.

    wx, wy
    """
    i0 = np.where((wx == 0) & (wy==0))[0]
    wx[i0] = 1.0
    wy[i0] = 1.0
    N = len(sxx)

    a = szz[:, None]
    b = sxz[:, None]*wx+syz[-1][:, None]*wy
    c = sxx[:, None]*wx*wx+2*sxy[:, None]*wx*wy+syy[:, None]*wy*wy

    aλ = np.sqrt(a*c-b*b)
    tanh = np.tanh(aλ[:-1] * thicknesses[:, None] / a[:-1])
    Q = np.empty((N, len(wx)))
    Q[-1] = 1/aλ[-1]
    for i in range(N-2, -1, -1):
        aλQ = aλ[i]*Q[i+1]
        Q[i] = (aλQ+tanh[i])/(aλ*(aλQ*tanh[i]+1))
    Q[:, i0] = 0.0
    Q /= (2*np.pi)

    # a = szz[-1]
    # b = sxz[-1]*wx+syz[-1]*wy
    # c = sxx[-1]*wx*wx+2*sxy[-1]*wx*wy+syy[-1]*wy*wy
    # aλ = np.sqrt(a*c-b*b)
    # Q = 1/(aλ)
    # for i in range(N-2, -1, -1):
    #     a = szz[i]
    #     b = sxz[i]*wx+syz[i]*wy
    #     c = sxx[i]*wx*wx+2*sxy[i]*wx*wy+syy[i]*wy*wy
    #     aλ = np.sqrt(a*c-b*b)
    #     λ = aλ/a
    #     tanh = np.tanh(thicknesses[i]*λ)
    #     aλQ = aλ*Q
    #     Q = (aλQ+tanh)/(aλ*(aλQ*tanh+1))
    # Q /= (2*np.pi)

    g_a = np.empty((N, len(wx)))
    g_b = np.empty((N, len(wx)))
    g_c = np.empty((N, len(wx)))
    Q_dh = np.empty((N-1, len(wx)))
    g_Q = 1
    for i in range(N-1):
        aλQ = aλ[i]*Q[i+1]
        g_aλQ = (1 - tanh[i]**2)/(aλ*(aλQ*tanh[i]+1)**2)*g_Q
        g_aλ = -Q[i]/aλ*g_Q
        g_tanh = (1 - aλQ**2)/(aλ*(aλQ*tanh[i]+1)**2)*g_Q
        g_aλ += Q[i+1]*g_aλQ
        g_Q = aλ[i]*g_aλQ
        g_λ = (1.0 - tanh[i]**2) * thicknesses[i] * g_tanh
        Q_dh[i] = (1.0 - tanh[i]**2) * aλ[i] / a[i] * g_tanh
        g_a[i] = -aλ[i]/(a[i]*a[i])*g_λ
        g_aλ += g_λ/a
        g_a[i] += 0.5*c[i] / aλ[i] * g_aλ
        g_b[i] = -b[i] / aλ[i] * g_aλ
        g_c[i] = 0.5 * a[i] / aλ[i] * g_aλ
    g_aλ = -2 / aλ[i] * g_Q
    g_a[-1] += 0.5 * c[-1] / aλ[-1] * g_aλ
    g_b[-1] = -b[-1] / aλ[-1] * g_aλ
    g_c[-1] = 0.5 * a[-1] / aλ[-1] * g_aλ

    Q_sxx = wx*wx * g_c
    Q_sxy = 2*wx*wy * g_c
    Q_syy = wy * wy * g_c
    Q_sxz = wx * g_b
    Q_syz = wy * g_b
    Q_szz = g_a
    return Q_sxx, Q_sxy, Q_sxz, Q_syy, Q_syz, Q_szz
