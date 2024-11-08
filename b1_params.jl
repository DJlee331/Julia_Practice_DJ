# Jesus Fernandez-Villaverde, Samuel Hurtado and Galo Nuno (2018)
# Financial Frictions and the Wealth Distribution


# ###########################################################
# # Model parameters
# ###########################################################

# const alpha         = 0.35;                       # Capital share   
# const delta         = 0.1;                        # Depreciation
# const gamma         = 2;                          # CRRA utility with parameter s
# const rho           = 0.05;                       # discount rate
# const Zmean         = 0.00;                       # mean for TFP
# const theta         = 0.50;                       # AR for TFP
# const sigma         = 0.01;                       # sigma for TFP
# const la1           = 0.5;                        # transition prob
# const la2           = 1/3;
# const la            = [la1,la2];                  # vector of transition prob
# const z1            = 0.93;                       # labor productivity
# const z2            = 1+ la2/la1 *(1- z1);
# const amin          = 0;                          # borrowing constraint
# const amax          = 30;                         # max value of assets
# const Kmin          = 3.5;                        # relevant range for aggregate K
# const Kmax          = 3.9;
# const Zmin          = -0.04;                      # relevant range for aggregate Z
# const Zmax          =  0.04;
      
# const nval_a        = 1001;                       # number of points between amin and amax, please make it end in 1
# const nval_z        = 2;                          # number of options for z (the idiosincratic shock)
# const nval_K        = 4;                          # number of points between Kmin and Kmax
# const nmul_K        = 5;                          # number of interpolated points to add between every contiguous pair of K values
# const nval_KK       = (nval_K-1)*nmul_K+1;        # number of points between Kmin and Kmax, in the interpolated version of A
# const nval_Z        = 41;                         # number of points between Zmin and Zmax, please make it end in 1
# const nval_Zsim     = 12000;                      # number of periods in simulation
# const delay_Zsim    = 1200;                       # number of initial periods in simulation that will not be used

# const dt       = 1/12;                            # size of t jump
# const da  = (amax-amin)/(nval_a-1);               # size of a jump
# const dK  = (Kmax-Kmin)/(nval_K-1);               # size of K jump
# const dKK = (Kmax-Kmin)/(nval_KK-1);              # size of K jump after interpolation in K
# const dZ  = (Zmax-Zmin)/(nval_Z-1);               # size of Z jump


###########################################################
# parameters for the algorithm
###########################################################

const multi_sim    =   4    ;                     # number of simulation starts (i.e. one plus number of times we stop the simulation and start again from the ss)
const delay_sim    =  500/dt;                     # number of initial periods in each simulation that will not be used (pre-heat)
const used_sim     = 5000/dt;                     # number of periods in each simulation that will actually be used
const each_sim     = delay_sim+used_sim;          # number of periods on each simulation
const nval_sim     = multi_sim*each_sim;          # total number of periods in all simulations
const rngseed1     = 123;                         # RNG seed for calculating the shocks for the simulation
                                            # there used to be a second seed for the mini-batch selection in the NN training, but we got rid of that by moving to batch gradient descent

const maxitHJB = 100;                             # max number of iterations for the HJB
const critHJB  = 10^(-6);                         # convergence crit for the HJB 
const weHJB    = 0.5;                             # relaxation algorithm for HJB
const Delta    = 1000;                            # Delta in HJB
const maxitPLM = 200;                             # max number of iterations of the full algorithm
const critPLM  = 0.00050;                         # convergence crit for determining that the PLM has converged
const wePLM    = 0.3;                             # Initial weigth in the relaxation algorithm for PLM convergence
const wePLM1   = 0.9;                             # reduction of the relaxation algorithm: wePLM = wePLM*wePLM1+wePLM2
const wePLM2   = 0.005;                           # reduction of the relaxation algorithm

###########################################################
# Neural network parameters
###########################################################
const network_width  =    16;                     # Number of neurons in the hidden layer
const mynoise        =     1;                     # Size of random initial NN parameters
const lambda         =   0.1;                     # NN regularization parameter
const NN_iters       = 10000;                     # Number of iterations to train the network
const NN_starts      =    10;                     # Number of random restarts of the network training (only affects the first step, then it just starts from the previous NN)
const learning_speed =  0.01;                     # Only affects the first step, then this becomes adaptative
const reglimY        =     4;                     # NN input normalization
const reglimX        =     4;                     # NN input normalization
###########################################################

using JLD2
@load "SS_result.jld2" K_ss g_ss A_ss

# VARIABLES: the main ones will be 4-dimmensional matrices
# of size (nval_a, nval_z, nval_K, nval_Z), with subscripts (ia, iz, iK, kZ)
# there will be lots of repeated information in them (many of these matrices
# will be flat in all dimensions except one), but I can afford it
# (with 51x2x61x41, each of these matrices takes slightly over 2 MB of memory)

a_grid   = collect(range(amin,amax,nval_a));           # 1D - assets
z_grid   = [z1,z2];                               # 1D - productivity
K_grid   = collect(range(Kmin,Kmax,nval_K));           # 1D - capital
KK_grid  = collect(range(Kmin,Kmax,nval_KK));          # 1D - capital, with interpolated values
Z_grid   = collect(range(Zmin,Zmax,nval_Z));           # 1D - TFP

a      = zeros(nval_a, nval_z, nval_K, nval_Z);
z      = zeros(nval_a, nval_z, nval_K, nval_Z);
K      = zeros(nval_a, nval_z, nval_K, nval_Z);
KK     = zeros(nval_a, nval_z, nval_KK,nval_Z);
Z      = zeros(nval_a, nval_z, nval_K, nval_Z);
ZZ     = zeros(nval_a, nval_z, nval_KK,nval_Z);
for iz=1:nval_z, iK=1:nval_K, iZ=1:nval_Z
            a[:,iz,iK,iZ]=a_grid;
 end
for ia=1:nval_a, iK=1:nval_K, iZ=1:nval_Z
    z[ia,:,iK,iZ] = z_grid;
end
for ia=1:nval_a, iz=1:nval_z, iZ=1:nval_Z
    K[ia,iz,:,iZ] = K_grid;
end
for ia=1:nval_a, iz=1:nval_z, iZ=1:nval_Z
    KK[ia,iz,:,iZ] =KK_grid;
end
for ia=1:nval_a, iz=1:nval_z, iK=1:nval_K
    Z[ia,iz,iK,:] = Z_grid;
end
for ia=1:nval_a, iz=1:nval_z, iK=1:nval_K
    ZZ[ia,iz,iK,:] = Z_grid;
end

KK_grid_2D = zeros(nval_KK, nval_Z);
Z_grid_2D = zeros(nval_KK, nval_Z);
for iZ=1:nval_Z
    KK_grid_2D[:,iZ] = KK_grid;
end
for iK=1:nval_KK
    Z_grid_2D[iK,:]=Z_grid;
end

a2 = a[:,:,1,1]; # this one is 2D instead of 4D, we need it for a simpler KFE algorithm

# Interest rates and wages (4D matrices that don't depend on anything but parameters) - WE ARE ASSUMING L=1
r =  α * K.^(α-1).*exp.(Z) .- δ;
w = (1-α) * K.^α .*exp.(Z);


