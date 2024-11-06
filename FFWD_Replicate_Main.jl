using DataFrames, CSV
using LinearAlgebra, SparseArrays
using Printf
# 1. b1_main.m 돌리고 ss_result를 가져와서
# 2. b1_params.m 돌려서 모수 값 박은 다음에
# 3. b2_Klm 돌려서 모형을 푼다
# 4. 그리고 c2_PLM_phase / c3_sss / c4_errors_zone / c5_IRF 돌려서 그림을 그린다

## 1. Compute steady state

#params
alpha = 0.35;                       # Capital share   
delta = 0.1;                        # Depreciation
gamma = 2;                          # CRRA utility with parameter s
rho = 0.05;                       # discount rate
la1 = 0.5;                        # transition prob
la2 = 1 / 3;
z1 = 0.93;                       # labor productivity
z2 = 1 + la2 / la1 * (1 - z1);
amin = 0;                          # borrowing constraFloat
amax = 30;                         # max value of assets

II = 1001;                            # numer of poFloats between amin and amax, please make it end in 1
maxit = 100;                             # max number of iterations
crit = 10^(-6);                         # convergence ctrit 
Delta = 1000;                            # Delta in HJB
maxitK = 400;                             # max number of iterations
critK = 10^(-3);                         # convergence ctrit
weK = 0.001;                           # Weigth in the relaxation algorithm

mutable struct Parameters
    alpha::Float64
    delta::Float64
    gamma::Float64
    rho::Float64
    la1::Float64
    la2::Float64
    z1::Float64
    z2::Float64
    amin::Float64
    amax::Float64
    I::Int64
    maxit::Int64
    crit::Float64
    Delta::Float64
    maxitK::Int64
    critK::Float64
    weK::Float64
end

params = Parameters(alpha, delta, gamma, rho, la1, la2, z1, z2, amin, amax, II, maxit, crit, Delta, maxitK, critK, weK)



K_guess = 3.69;


struct mdl_result
    a::Vector
    aa::Matrix
    z::Vector
    zz::Matrix
    V::Matrix
    A::Matrix
    c::Matrix
    r::Float64
    w::Float64
    da::Float64
    dV::Matrix
end

function b4_KFE_stationary(params, results)
    # This codes computes the KFE equation in a Bewley economy with nominal
    # long-term debt
    # Developped by Nu� and Thomas (2015) based on codes by Ben Moll 
    I = params.I                # numer of poFloats
    da = results.da
    AT = copy(results.A')
    b = zeros(2 * I, 1)

    # need to fix one value, otherwise matrix is singular
    i_fix = 4
    b[i_fix] = 0.1

    row = hcat(zeros(1, i_fix - 1), 1, zeros(1, 2 * I - i_fix))
    AT[i_fix, :] = row

    # Solve linear system
    gg = AT \ b
    g_sum = gg' * ones(2 * I, 1) * da
    gg = gg ./ g_sum

    g = hcat(gg[1:I], gg[I+1:2*I])

    dist1 = g
    return dist1
end

function b3_HJB_stationary(params, K)
    alpha = params.alpha
    delta = params.delta
    gamma = params.gamma            # CRRA utility with parameter s
    rho = params.rho              # discount rate
    z1 = params.z1               # labor productivity
    z2 = params.z2
    la1 = params.la1              # transition prob
    la2 = params.la2
    I = params.I                # numer of poFloats
    amin = params.amin             # borrowing constraFloat
    amax = params.amax             # max value of assets
    maxit = params.maxit            # max number of iterations
    crit = params.crit             # convergence ctrit 
    Delta = params.Delta            # Delta in HJB

    # VARIABLES
    a = collect(range(amin, amax, Int(I)))               # assets 1D  - I elements
    da = (amax - amin) / (I - 1)
    z = [z1, z2]                             # product 1D - 2 elements
    la = [la1, la2]
    aa = repeat(a, 1, 2)                                # assets 2D
    zz = ones(I, 1) * z'                          # product 2D
    dVf = zeros(I, 2)
    dVb = zeros(I, 2)
    c = zeros(I, 2)

    Aswitch = vcat(hcat(-sparse(LinearAlgebra.I(I)) * la1, sparse(LinearAlgebra.I(I)) * la1), hcat(sparse(LinearAlgebra.I(I)) * la2, -sparse(LinearAlgebra.I(I)) * la2))

    #  Floaterest rates and wages
    r = alpha * K^(alpha - 1) - delta
    w = (1 - alpha) * K^alpha
    # INITIAL GUESS
    v0 = zeros(I, 2)
    v0[:, 1] = ((w .* z[1] .+ r .* a) .^ (1 - gamma)) / (1 - gamma) / rho
    v0[:, 2] = ((w .* z[2] .+ r .* a) .^ (1 - gamma)) / (1 - gamma) / rho
    v = v0

    # Begin Main iteration
    dist_result = zeros(maxit, 1)
    V = v0
    V_n = repeat(V, 1, 1, maxit)
    dV = zeros(size(V))
    A = sparse(zeros(2 * I, 2 * I))
    for n = 1:maxit
        V = v
        V_n[:, :, n] = V
        # Forward difference
        dVf[1:I-1, :] = (V[2:I, :] .- V[1:I-1, :]) / da
        dVf[I, :] = (w .* z .+ r .* amax) .^ (-gamma)    # will never be used, but impose state constraFloat a<=amax just in case

        # backward difference
        dVb[2:I, :] = (V[2:I, :] .- V[1:I-1, :]) / da
        dVb[1, :] = (w .* z .+ r .* amin) .^ (-gamma)    # state constraFloat boundary condition

        I_concave = dVb .> dVf   # indicator whether value function is concave (problems arise if this is not the case)

        # consumption and savings with forward difference
        cf = dVf .^ (-1 / gamma)
        ssf = w .* zz + r .* aa - cf
        # consumption and savings with backward difference
        cb = (dVb) .^ (-1 / gamma)
        ssb = w .* zz + r .* aa - cb
        # consumption and derivative of value function at steady state
        c0 = w .* zz + r .* aa
        dV0 = c0 .^ (-gamma)


        # dV_upwind makes a choice of forward or backward differences based on the sign of the drift    
        If = (ssf .> 0) .+ 0 # positive drift --> forward difference
        Ib = (ssb .< 0) .+ 0 # negative drift --> backward difference
        I0 = (-If - Ib .+ 1) # at steady state
        # make sure backward difference is used at amax
        # Ib(I,:) = 1; If(I,:) = 0;

        dV_Upwind = dVf .* If + dVb .* Ib + dV0 .* I0 # important to include third term
        c = (dV_Upwind) .^ (-1 / gamma)
        u = ((c .^ (1 - gamma) .- 1)) / (1 - gamma)

        # CONSTRUCT MATRIX A
        elem_a = -min.(ssb, 0) / da
        elem_e = max.(ssf, 0) / da
        elem_b = -elem_a - elem_e
        #   elem_b            = ( -ssf.*If + ssb.*Ib)/da;  # - max(ssf,0)/da + min(ssb,0)/da;

        A1 = sparse(diagm(elem_b[:, 1])) + spdiagm(-1 => elem_a[2:I, 1]) + spdiagm(1 => elem_e[1:I-1, 1])
        A2 = sparse(diagm(elem_b[:, 2])) + spdiagm(-1 => elem_a[2:I, 2]) + spdiagm(1 => elem_e[1:I-1, 2])
        A = vcat(hcat(A1, sparse(zeros(I, I))), hcat(sparse(zeros(I, I)), A2)) + Aswitch
        B = (rho + 1 / Delta) * sparse(LinearAlgebra.I(2 * I)) - A

        u_stacked = vec(u)
        V_stacked = vec(V)

        b = u_stacked + V_stacked / Delta
        V_stacked = B \ b  # SOLVE SYSTEM OF EQUATIONS
        V = hcat(V_stacked[1:I], V_stacked[I+1:2*I])

        dV = V - v
        v = V

        # CHECK CONVERGENCE
        dist_result[n] = maximum(abs.(dV))
        if dist_result[n] < crit
            println("Value Function Converged, Iteration = " * string(n))
            break
        end
    end
    # results=mdl_result(a,aa,z,zz,V,A,c,r,w,da,dV)
    results = mdl_result(a, aa, z, zz, V, A, c, r, w, da, dV)
    return results
end


function b2_equilibrium(params, K)
    # Computes the excess demand
    maxitK = params.maxitK
    critK = params.critK
    weK = params.weK
    n = 1
    for n = 1:maxitK
        results = b3_HJB_stationary(params, K)
        distribution = b4_KFE_stationary(params, results)
        S = sum(sum(results.aa .* distribution * results.da))
        if abs(S - K) < critK
            @printf "K =%.2f \n" K
            break
        else
            K = weK * S + (1 - weK) * K
            # println("K = ",K)
        end
    end

    if n == maxitK
        println("ERROR: max iterations reached")
    end
    return K
end



K = b2_equilibrium(params, K_guess);
results = b3_HJB_stationary(params, K);
dist_result = b4_KFE_stationary(params, results);

## MOMENTS
rnge = findlast(x -> x < 0, results.a);
S = sum(results.aa .* dist_result * results.da);
C = sum(results.c .* dist_result .* results.da)
L = sum(results.zz .* dist_result .* results.da)
Y = S^alpha;
println("Total capital (% GDP)")
@printf "%.2f" S / Y * 100
println("Total labor")
@printf "%.2f" L
println("Total output")
println(Y)
println("Interest rate(%)")
@printf "%.2f" results.r * 100

## Plot
alimmax = amax/Y*0.8;
using Plots, LaTeXStrings
p=plot(layout=(2,2),background=:white,background_outside=:lightgray)
plot!(p[1],results.a/Y,results.V[:,1],lw=3,color=:black,legend=false,title=L"Value Function, $v(a)$",xlim=(amin/Y, alimmax))
plot!(p[1],results.a/Y,results.V[:,2],lw=3,color=:red,legend=false,linestyle=:dash)

plot!(p[2],results.a/Y,results.c[:,1],lw=3,color=:black,legend=false,title=L"Consumption, $c(a)$",xlim=(amin/Y, alimmax))
plot!(p[2],results.a/Y,results.c[:,2],lw=3,color=:red,legend=false,linestyle=:dash)

s = results.aa*results.r + results.w*results.zz - results.c;
plot!(p[3],results.a/Y,s[:,1],lw=3,color=:black,legend=false,title=L"Drift, $s(a)$",xlim=(amin/Y, alimmax))
plot!(p[3],results.a/Y,s[:,2],lw=3,color=:red,legend=false,linestyle=:dash)


plot!(p[4],results.a/Y,dist_result[:,1],lw=3,color=:black,legend=true,title=L"distribution, $f(a)$",xlim=(amin/Y, alimmax),label="State 1")
plot!(p[4],results.a/Y,dist_result[:,2],lw=3,color=:red,linestyle=:dash,label="State 2")

xaxis!(L"Assets, $a$")

savefig("Stead_State.png")

K_ss = S
g_ss = dist_result
A_ss = results.A

using JLD2
@save "SS_result.jld2" K_ss g_ss A_ss


##
