import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.experimental.numpy as tnp

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#tf.config.run_functions_eagerly(True) # needed for numpy compatibility in tf.functions. Otherwise BUG in make_GP_SIM when avluating theta, which becomes of class 'tensorflow.python.framework.ops.Tensor' not class 'tensorflow.python.framework.ops.EagerTensor' during MCMC sampling.

tfb = tfp.bijectors
tfd = tfp.distributions
psd_kernels = tfp.math.psd_kernels

#### SET MODEL TO RUN ####
# (0) higdon, (1) koh, (2) ml, (3) f
IND_MODEL = 0

# FULL_BI = 0, 1 bool. parameter for def. of gp_model_conditioned
FULL_BI = True

#
LOGSCALE_X = True

logscale_x = LOGSCALE_X

####################
# DATA PREPARATION #
####################

NbDimX = 1
NbDimY = 1
NbDimT = 2

observation_noise_variance = 0.0025
observation_noise_variance_SIM = observation_noise_variance*1e-0

############ EXPERIMENTAL MODEL - DATA ############

# INPUT SPACE (OBS, PRED)
#NbObs_EXP = 6
NbPred = 100 # serves also for nb of sim. predictions

# generate Xobs for exp. observations (test points must be of shape (NbObs_EXP, NbDimX))
# data set 1 (SW)
#XObs_EXP = tnp.array((2.0000,2.6000,3.3800,4.3940,5.7120,7.4260))[..., tnp.newaxis]#9.6540,12.5500,16.3150,21.2090))[..., tnp.newaxis]
#R_Die = 1.
#YObs_EXP = tnp.array((2.7587,2.8780,2.9404,2.9918,3.0718,3.1419)) / (2.*R_Die) #3.0958,3.0669,3.0660,3.1023)) / (2.*R_Die) # SW
# data set 2 (DynSW)
data_X = tnp.array((2.0000,2.6000,3.3800,4.3940,5.7120,7.4260,9.6540,12.5500,16.3150))[..., tnp.newaxis]
if LOGSCALE_X:
    data_X = tnp.log(data_X)

R_Die = 1.
data_Y = tnp.array((2.7055,2.8726,2.9582,2.9994,2.9851,3.1117,3.1309,3.1855,3.2369)) / (2.*R_Die) #3.0629,2.6983,2.5239,2.5141,2.6856,2.3131,2.2846 / (2.*R_Die) # DynSW

n = len(data_X)
itrain = np.ones(n,np.bool_)
itrain[slice(int((n-1)/2-1),n,3)] = 0
itrain[-1] = 0

XObs_EXP = data_X[itrain]
YObs_EXP = data_Y[itrain]

XTest_EXP = data_X[~itrain]
YTest_EXP = data_Y[~itrain]

#XTest_EXP
#YTest_EXP

NbObs_EXP = len(XObs_EXP)

ax = min(XObs_EXP)[0]*0.5
bx = max(XObs_EXP)[0]*1.2

xx = tnp.linspace(
    ax, bx, num=NbPred-len(XTest_EXP))[..., tnp.newaxis]
XPred_EXP = tnp.sort(tnp.concatenate(
    [XTest_EXP, xx], axis=0), axis=0)
X_EXP = tnp.concatenate(
    [XObs_EXP, XPred_EXP], axis=0)

sliceObs_EXP = slice(0, NbObs_EXP)
slicePred_EXP = slice(NbObs_EXP, NbObs_EXP + NbPred)


############ SIMULATION MODEL - DATA ############
#
# SIMULATION MODEL INPUT SPACE (OBS(X,.))
#
# initialize Xobs0 for sim. extended exp. observations ('training?' points must be of shape (NbObs_EXP, (NbDimX + NbDimT)))
# the theta entries will change during execution (call to make_GP_SIM) and have to be initialized by zero
xx = tnp.zeros((NbObs_EXP,NbDimT), dtype=tnp.float64)
XObs0_SIM = tnp.concatenate([XObs_EXP, xx], axis=1)
#
# SIMULATION MODEL INPUT SPACE (OBS(X,T))
#
def make_prior_theta():
    alpha = tnp.array((2.,2.))
    beta = tnp.array((2.,2.))
    shift = tnp.array((0.,0.))
    scale=tnp.array((2.,.2))
    pdf = tfd. Independent(
        tfb.Chain(
            [ tfb.Shift(shift=shift), tfb.Scale(scale=scale) ]
        ) ( tfd.Beta(alpha,beta) ),
        reinterpreted_batch_ndims=1, name="theta") # theta = (llambda, beta)
    pdf.record={'dist': 'Beta',
                'params': {'alpha': alpha, 'beta': beta, 'shift': shift, 'scale': scale},
                'mean': pdf.mean(),
                'std': pdf.stddev()}
    return pdf

# calculate tfd.Beta.quantile (inverse of tfd.Beta.cdf; NotImplementedError) from scipy.stats
def Beta_quantile(q, dist_Beta):
    alpha = dist_Beta.record["params"]["alpha"].numpy()
    beta = dist_Beta.record["params"]["beta"].numpy()
    shift = dist_Beta.record["params"]["shift"].numpy()
    scale = dist_Beta.record["params"]["scale"].numpy()
    return tnp.array(scipy.stats.beta.ppf(q, alpha, beta, loc=shift, scale=scale))

# create mesh
cartesian_XObs_SIM = False
if cartesian_XObs_SIM:
    NbObs_per_dim_SIM = 5
    NbObs_SIM = NbObs_per_dim_SIM ** (NbDimX + NbDimT)
    # generate Xobs for sim. observations (training points must be of shape (NbObs_SIM, (NbDimX + NbDimT)))
    dist_t = make_prior_theta()
    mean_t = dist_t.mean()
    std_t = dist_t.stddev()
    at =  mean_t-2*std_t # Gaussian quantiles
    bt =  mean_t+2*std_t # Gaussian quantiles
    ax = min(XPred_EXP)[0]
    bx = max(XPred_EXP)[0]
    aa = tnp.array([ax, at[0], at[1]], dtype=tnp.float64)  # [shearRate, llambda, beta] (min)
    bb = tnp.array([bx, bt[0], bt[1]], dtype=tnp.float64)  # [shearRate, llambda, beta] (max)
    xx = tf.transpose( tnp.linspace(aa, bb, NbObs_per_dim_SIM) )
    xxx = tnp.meshgrid(*xx)
    XObs_SIM = tf.transpose( tf.reshape( tnp.array(xxx), (len(a_obs),-1) ) )
else:
    n_obs_x = 10
    n_obs_t = 20
    # t-space (LH mesh)
    lhs = scipy.stats.qmc.LatinHypercube(d=NbDimT, seed=42).random(n_obs_t)
    print('LHS quality', scipy.stats.qmc.discrepancy(lhs))
    dist_t = make_prior_theta()
    #mean_t = dist_t.mean().numpy()
    #std_t = dist_t.stddev().numpy()
    #l_bounds =  mean_t-2*std_t # Gaussian quantiles
    #u_bounds =  mean_t+2*std_t # Gaussian quantiles
    #lhs = scipy.stats.qmc.scale(lhs, l_bounds, u_bounds)
    #lhs = tnp.array(lhs, dtype=tnp.float64)
    lhs = Beta_quantile(lhs, dist_t)
    # x-space (regular mesh)
    ax = min(XPred_EXP)[0][...,np.newaxis]
    bx = max(XPred_EXP)[0][...,np.newaxis]
    xx = tnp.linspace(ax, bx, n_obs_x)
    # full (x,t)-space mesh
    XObs_SIM = tnp.concatenate(( np.kron(xx, np.ones((len(lhs),1))),
                                np.kron(np.ones_like(xx), lhs)
                               ), axis=-1 )
    NbObs_SIM = len(XObs_SIM)    

#
# SIMULATION MODEL INPUT SPACE (PRED)
#
# initialize test points Xgp for sim. prediction (test points must be of shape (NbPred, (NbDimX + NbDimT)))
# the theta entries will change during execution (call to make_GP_SIM) and have to be initialized by zero
xx = tnp.zeros((NbPred,NbDimT), dtype=tnp.float64)
XPred_SIM = tnp.concatenate((XPred_EXP, xx),axis=1)

# concat Xobs and Xgp
X_SIM = tnp.concatenate(
    [XObs0_SIM, XObs_SIM, XPred_SIM], axis=0)

sliceObs_SIM = slice(NbObs_EXP,
                     NbObs_EXP + NbObs_SIM)
slicePred_SIM = slice(NbObs_EXP + NbObs_SIM,
                      NbObs_EXP + NbObs_SIM + NbPred)

# simulator
def f_sim(x, t, LOGSCALE_X=LOGSCALE_X):
    shearRate = x[...,0]
    if LOGSCALE_X:
        shearRate = tnp.exp(shearRate) # if semilog scale
    llambda = tf.reshape( t[...,0], (-1,1))
    beta = tf.reshape( t[...,1], (-1,1))
    #llambda = t[..., tnp.newaxis, 0]
    #beta = t[..., tnp.newaxis, 1]
    Sw = (1-beta)*llambda*shearRate
    return (1+Sw**2/2)**(1/6) + .13

YObs_SIM = tf.reshape(
    f_sim(XObs_SIM[..., tnp.newaxis, 0:NbDimX], XObs_SIM[..., NbDimX:NbDimX+NbDimT]), -1)

############ JOINT MODEL - DATA ############

ZObs = tnp.concatenate([YObs_EXP,YObs_SIM])


##########################
# PORBABILISTIC MODELING #
##########################


############ EXPERIMENTAL MODEL - MD GP DISTRIBUTION ############

def make_GP_MD(length_scale_MD, amplitude_MD, mean_MD):
    # NOTE: loc == None implies zero mean.
    jitter = 1E-5

    # TODO: Implement manually.
    kernel = psd_kernels.ExponentiatedQuadratic(amplitude_MD, length_scale_MD)

    kernel_matrix = kernel.matrix(
        X_EXP, X_EXP)
    # Add jitter
    kernel_matrix = tf.linalg.set_diag(
        kernel_matrix, tf.linalg.diag_part(kernel_matrix) + jitter)

    scale = tf.linalg.LinearOperatorLowerTriangular(
        tf.linalg.cholesky(kernel_matrix),
        is_non_singular=True)

    loc = tnp.ones((X_EXP.shape[0])) * mean_MD

    gp = tfp.distributions.MultivariateNormalLinearOperator(
        loc=loc,
        scale=scale,
        validate_args=True,
        allow_nan_stats=False,
        name="GP_MD")

    return gp

############ SIMULATION MODEL - SIM GP DISTRIBUTION ############

def make_GP_SIM(theta, length_scale_SIM, amplitude_SIM, mean_SIM):#, X_SIM=X_SIM): 
    # W: ensure that amplitude.dtype = length_scale = theta.dtype
    # NOTE: loc == None implies zero mean.
    jitter = 1E-5

    # TODO: Implement manually.
    kernel = psd_kernels.ExponentiatedQuadratic(amplitude_SIM, length_scale=1.) #tf.ones_like(amplitude_SIM))

    kernel = tfp.math.psd_kernels.FeatureScaled(kernel, length_scale_SIM)

    ############## WORKS ONLY, IF EAGER MODE SWITCHED ON
    ## # create batch_shape copies of X_SIM AND fill theta values to enriched exp. abscissas
    ## O=tnp.ones( kernel.batch_shape + X_SIM.shape )
    ## X_SIM_GP = O*X_SIM # X_SIM_GP.shape=(batch_shape, n+m, 3), theta.shape=(batch_shape, 2)
    ## X_SIM_GP[..., sliceObs_EXP, 1:3] = theta[..., tnp.newaxis, :] # works for feature_ndim=1 and any batch_shape
    ###############

    # create batch_shape copies of X_SIM AND fill theta values to enriched exp. abscissas
    # NOTE: Have to avoid numpy calculations with tf.Tensors, since tf.function calls have eager mode turned off, for performance reasons.
    #O=tnp.ones( kernel.batch_shape + X_SIM.shape )
    #X_SIM_GP = O*X_SIM # X_SIM_GP.shape=(batch_shape, X_SIM.shape=(n+m, 3)), theta.shape=(batch_shape, theta.event_shape=(1,2))

    diag = tnp.ones((NbDimT))
    b=tnp.zeros((NbDimT,NbDimX+NbDimT)) # shape=(theta.event_shape, X_SIM.shape[1]), s.t. ((theta*c)@b).shape=(1, X_SIM.shape[1])
    b = tf.linalg.set_diag(b,diag,k=1)
    c = np.ones((1, NbDimT)) # s.t. (theta*c).shape = (1,theta.shape)
    t=(theta*c)@b # tf.Tensor: t.shape=(1, X_SIM.shape[1])

    #O=np.zeros(X_SIM.shape)
    #O[sliceObs_EXP, NbDimX:NbDimX+NbDimT] = 1. # Higdon: sim. regression on exp. observations take rv theta as abscissas
    #O[slicePred_SIM, NbDimX:NbDimX+NbDimT] = 1. # predictions also take rv theta as abscissas

    A=tnp.zeros((NbObs_EXP,NbDimX))
    B=tnp.ones((NbObs_EXP,NbDimT))
    AB=tnp.concatenate((A,B),axis=1)
    C=tnp.zeros((NbObs_SIM,NbDimX+NbDimT))
    D=tnp.zeros((NbPred,NbDimX))
    E=tnp.ones((NbPred,NbDimT))
    DE=tnp.concatenate((D,E),axis=1)
    O=tnp.concatenate((AB,C,DE),axis=0)

    
    O=O*t # tf.Tensor
    X_SIM_GP = X_SIM + O # X_SIM_GP.shape=X_SIM.shape=(n+m, NbDimX+NbDimT)

    kernel_matrix = kernel.matrix(
        X_SIM_GP, X_SIM_GP)

    #mean_SIM = tnp.mean(YObs_SIM)
    loc = tnp.ones((X_SIM_GP.shape[0])) * mean_SIM
    
    # Add jitter
    kernel_matrix = tf.linalg.set_diag(
        kernel_matrix, tf.linalg.diag_part(kernel_matrix) + jitter)

    scale = tf.linalg.LinearOperatorLowerTriangular(
        tf.linalg.cholesky(kernel_matrix),
        is_non_singular=True)

    gp = tfp.distributions.MultivariateNormalLinearOperator(
        loc=loc,
        scale=scale,
        validate_args=True,
        allow_nan_stats=False,
        name="GP_SIM")

    return gp


############ (HYPER-)PARAMETER DISTRIBUTIONS ############
aXObs_SIM = tnp.min(XObs_SIM, axis=0)
bXObs_SIM = tnp.max(XObs_SIM, axis=0)
aYObs_SIM = tnp.min(YObs_SIM, axis=0)
bYObs_SIM = tnp.max(YObs_SIM, axis=0)
meanYObs_SIM = tnp.mean(YObs_SIM, axis=0)
stdYObs_SIM = tnp.std(YObs_SIM, axis=0)

aXObs_EXP = tnp.min(XObs_EXP, axis=0)
bXObs_EXP = tnp.max(XObs_EXP, axis=0)
aYObs_EXP = tnp.min(YObs_EXP, axis=0)
bYObs_EXP = tnp.max(YObs_EXP, axis=0)
meanYObs_EXP = tnp.mean(YObs_EXP, axis=0)
stdYObs_EXP = tnp.std(YObs_EXP, axis=0)

scale_x_mean =  tnp.array((.25, .25, 5.), dtype=tnp.float64) # wrt. (bXObs_SIM-aXObs_SIM): (x, llambda, beta)
scale_x_low =  scale_x_mean * tnp.array((.5, .5, .5), dtype=tnp.float64) # (x, llambda, beta)
scale_x_high =  scale_x_mean * tnp.array((2., 2., 2.), dtype=tnp.float64) # (x, llambda, beta)

scale_y_mean =  tnp.array((.4), dtype=tnp.float64) # wrt. YObs__
scale_y_low =  scale_y_mean * tnp.array((.5), dtype=tnp.float64)
scale_y_high =  scale_y_mean * tnp.array((1.5), dtype=tnp.float64)

def make_InverseGamma_params(mean, stddev):
    alpha = mean**2/stddev**2+tnp.float64(2)
    beta = mean*(alpha-1)
    return alpha, beta


def make_prior_mean_MD():
    # if model is MD
    loc = tnp.float64(0.)
    scale = stdYObs_EXP
    low = -bYObs_EXP
    high = bYObs_EXP
    if IND_MODEL==2: # if model is ML (pure GP)
        loc = meanYObs_EXP
        scale = stdYObs_EXP
        low = aYObs_EXP
        high = bYObs_EXP
    pdf = tfd.Independent( tfd.TruncatedNormal(loc=loc, scale=scale, low=low, high=high), reinterpreted_batch_ndims=0, name="mean_MD" )
    pdf.record={
        'dist': pdf.distribution.name,
        'params': {'mean': pdf.mean(), 'stddev': pdf.stddev()},
        'mean': pdf.mean(),
        'std': pdf.stddev()}
    return pdf

def make_prior_amplitude_MD():
    mean = stdYObs_EXP
    std = tnp.abs( (bYObs_EXP-aYObs_EXP)/2. - mean )
    alpha, beta = make_InverseGamma_params(mean, std)
    pdf = tfd.Independent( tfd.InverseGamma(concentration=alpha, scale=beta), reinterpreted_batch_ndims=0, name="amplitude_MD")
    pdf.record={
        'dist': pdf.distribution.name,
        'params': {'alpha': alpha, 'beta': beta},
        'mean': pdf.mean(),
        'std': pdf.stddev()}
    return pdf

def make_prior_length_scale_MD():
    L_mean = ( (bXObs_SIM-aXObs_SIM) * scale_x_mean )[0]
    low = ( (bXObs_SIM-aXObs_SIM) * scale_x_low )[0]
    high = ( (bXObs_SIM-aXObs_SIM) * scale_x_high )[0]
    L_std = (high-low)/4.
    pdf = tfd.Independent( tfd.TruncatedNormal(loc=(high-low)/2., scale=L_std, low=low, high=high), reinterpreted_batch_ndims=0, name="length_scale_MD" )
    pdf.record={
        'dist': pdf.distribution.name,
        #'params': {'low': low, 'high': high},
        'params': {'mean': pdf.mean(), 'stddev': pdf.stddev()},
        'mean': pdf.mean(),
        'std': pdf.stddev()}
    return pdf

def make_prior_mean_SIM():
    loc = meanYObs_SIM
    scale = stdYObs_SIM
    low = aYObs_SIM
    high = bYObs_SIM
    pdf = tfd.Independent( tfd.TruncatedNormal(loc=loc, scale=scale, low=low, high=high), reinterpreted_batch_ndims=0, name="mean_SIM" )
    pdf.record={
        'dist': pdf.distribution.name,
        'params': {'mean': pdf.mean(), 'stddev': pdf.stddev()},
        'mean': pdf.mean(),
        'std': pdf.stddev()}
    return pdf

def make_prior_amplitude_SIM():
    mean = stdYObs_SIM
    std = tnp.abs( (bYObs_SIM-aYObs_SIM)/2. - mean )
    alpha, beta = make_InverseGamma_params(mean, std)
    pdf = tfd.Independent( tfd.InverseGamma(concentration=alpha, scale=beta), reinterpreted_batch_ndims=0, name="amplitude_SIM")
    pdf.record={
        'dist': pdf.distribution.name, # dist. underlying the pdf
        'params': {'alpha': alpha, 'beta': beta},
        'mean': pdf.mean(),
        'std': pdf.stddev()}
    return pdf

def make_prior_length_scale_SIM():
    L_mean = (bXObs_SIM-aXObs_SIM) * scale_x_mean
    low = (bXObs_SIM-aXObs_SIM) * scale_x_low 
    high = (bXObs_SIM-aXObs_SIM) * scale_x_high
    L_std = (high-low)/4.
    pdf = tfd.Independent( tfd.TruncatedNormal(loc=(high-low)/2., scale=L_std, low=low, high=high), reinterpreted_batch_ndims=1, name="length_scale_SIM" )
    pdf.record={'dist': pdf.distribution.name, # dist. underlying the pdf
                #'params': {'low': low, 'high': high},
                'params': {'mean': pdf.mean(), 'stddev': pdf.stddev()},
                'mean': pdf.mean(),
                'std': pdf.stddev()}
    return pdf



############ JOINT MODEL - (FULL JOINT) POSTERIOR DISTRIBUTION ############
    
# define projection tensors for the likelihood data GP_MD and GP_SIM: setting prediction output to zero AND inclusion mapping into the correct observation space ~ {ZObs}
#indDiag_EXP = tnp.arange(NbObs_EXP)
O_EXP = tnp.zeros((NbObs_EXP+NbPred, NbObs_EXP+NbObs_SIM))
diag=tf.concat((tf.ones(NbObs_EXP, dtype=tf.float64),tf.zeros(min(NbPred,NbObs_SIM), dtype=tf.float64)),axis=0)
O_EXP = tf.linalg.set_diag(O_EXP,diag)
#O_EXP[indDiag_EXP, indDiag_EXP] = 1
#o_shape_EXP = tnp.ones((1,NbObs_EXP+NbPred)) # needed for multiplication with GP_MD, to ensure broadcast compatibility with @O_EXP
#indDiag_SIM = tnp.arange(NbObs_EXP+NbObs_SIM)
O_SIM = tnp.zeros((NbObs_EXP+NbObs_SIM+NbPred, NbObs_EXP+NbObs_SIM))
diag=tf.ones(NbObs_EXP+NbObs_SIM, dtype=tf.float64)
O_SIM = tf.linalg.set_diag(O_SIM,diag)
#O_SIM[indDiag_SIM, indDiag_SIM] = 1
#o_shape_SIM = tnp.ones((1,NbObs_EXP+NbObs_SIM+NbPred)) # needed for multiplication with GP_SIM, to ensure broadcast compatibility with @O_SIM

# anisotropic noise: iid noise variance for exp obs and iid noise variance for sim obs:
observation_noise_diag_std = tnp.sqrt(observation_noise_variance)*tnp.concatenate((tnp.ones(NbObs_EXP),tnp.zeros(NbObs_SIM))) + tnp.sqrt(observation_noise_variance_SIM)*tnp.concatenate((tnp.zeros(NbObs_EXP),tnp.ones(NbObs_SIM)))

# define full joint PDF
gp_model_HIG = tfd.JointDistributionSequentialAutoBatched( [
    make_prior_mean_MD,
    make_prior_amplitude_MD,
    make_prior_length_scale_MD,
    make_GP_MD,
    #lambda length_scale_MD, amplitude_MD
    make_prior_mean_SIM,
    make_prior_amplitude_SIM,
    make_prior_length_scale_SIM,
    make_prior_theta,
    make_GP_SIM,
    lambda GP_SIM, theta, length_scale_SIM, amplitude_SIM, mean_SIM, GP_MD: tfd.MultivariateNormalDiag(
        #loc=( tf.squeeze((o_shape_EXP*GP_MD)@O_EXP + (o_shape_SIM*GP_SIM)@O_SIM )) # works in MCMC with nchain=1
        #loc=( tf.reshape( (o_shape_EXP*GP_MD)@O_EXP + (o_shape_SIM*GP_SIM)@O_SIM, -1 ) )
        #loc=( tf.matmul(o_shape_EXP*GP_MD,O_EXP)+tf.matmul(o_shape_SIM*GP_SIM,O_SIM) )
        loc=( tnp.dot(GP_MD,O_EXP)+tnp.dot(GP_SIM,O_SIM) ) # works in MCMC with nchain=1
        , scale_diag=observation_noise_diag_std,
        name="observations")],validate_args=True )

# Condition full joint posterior on observations
if FULL_BI:
    gp_model_conditioned_HIG = gp_model_HIG.experimental_pin(observations=ZObs)
else:
    L_SIM = (bXObs_SIM-aXObs_SIM)  * tnp.array((scale_x,scale_l,scale_b)) # [shearRate, llambda, beta]
    L_MD = L_SIM[0]
    #gp_model_conditioned_HIG = gp_model_HIG.experimental_pin(observations=ZObs, length_scale_MD=L_MD, length_scale_SIM=L_SIM)
    A_SIM = tnp.std(ZObs)
    A_MD = tnp.std(ZObs)
    gp_model_conditioned_HIG = gp_model_HIG.experimental_pin(observations=ZObs,
                                                             length_scale_MD=L_MD,
                                                             amplitude_MD=A_MD,
                                                             length_scale_SIM=L_SIM,
                                                             amplitude_SIM=A_SIM)


#### KOH
# define full joint PDF
o_EXP = tnp.zeros((NbObs_EXP+NbPred, NbObs_EXP))
o_diag=tf.concat((tf.ones(NbObs_EXP, dtype=tf.float64)),axis=0)
o_EXP = tf.linalg.set_diag(o_EXP,o_diag)

gp_model_KOH = tfd.JointDistributionSequentialAutoBatched( [
    make_prior_mean_MD,
    make_prior_amplitude_MD,
    make_prior_length_scale_MD,
    make_GP_MD,
    make_prior_theta,
    lambda theta, GP_MD: tfd.MultivariateNormalDiag(
        loc=( tnp.squeeze(tnp.dot(GP_MD,o_EXP)+tnp.dot(f_sim(X_EXP,theta),o_EXP)) ),
        scale_diag=tnp.sqrt(observation_noise_variance)*tnp.ones(NbObs_EXP),
        name="observations")], validate_args=True )

# Condition full joint posterior on observations
#gp_model_conditioned_KOH = gp_model_KOH.experimental_pin(observations=tf.reshape(YObs_EXP,(-1,1)))
if FULL_BI:
    gp_model_conditioned_KOH = gp_model_KOH.experimental_pin(observations=YObs_EXP)
else:
    L_MD = (bXObs_SIM-aXObs_SIM)[0] * scale_x
    #gp_model_conditioned_KOH = gp_model_KOH.experimental_pin(observations=YObs_EXP, length_scale_MD=L_MD)
    A_MD = tnp.std(ZObs) # tnp.std(YObs_EXP)
    gp_model_conditioned_KOH = gp_model_KOH.experimental_pin(observations=YObs_EXP,
                                                             length_scale_MD=L_MD,
                                                             amplitude_MD=A_MD)


#### PURE ML
# define full joint PDF
o_EXP = tnp.zeros((NbObs_EXP+NbPred, NbObs_EXP))
o_diag=tf.concat((tf.ones(NbObs_EXP, dtype=tf.float64)),axis=0)
o_EXP = tf.linalg.set_diag(o_EXP,o_diag)

gp_model_ML = tfd.JointDistributionSequentialAutoBatched( [
    make_prior_mean_MD,
    make_prior_amplitude_MD,
    make_prior_length_scale_MD,
    make_GP_MD,
    lambda GP_MD: tfd.MultivariateNormalDiag(
        loc=( tnp.squeeze(tnp.dot(GP_MD,o_EXP)) ),
        scale_diag=tnp.sqrt(observation_noise_variance)*tnp.ones(NbObs_EXP),
        name="observations")], validate_args=True )

# Condition full joint posterior on observations
if FULL_BI:
    gp_model_conditioned_ML = gp_model_ML.experimental_pin(observations=YObs_EXP)
else:
    L_MD = (bXObs_SIM-aXObs_SIM)[0] * scale_x
    #gp_model_conditioned_ML = gp_model_ML.experimental_pin(observations=YObs_EXP, length_scale_MD=L_MD)
    A_MD = tnp.std(ZObs) # tnp.std(YObs_EXP)
    gp_model_conditioned_ML = gp_model_ML.experimental_pin(observations=YObs_EXP,
                                                           length_scale_MD=L_MD,
                                                           amplitude_MD=A_MD)


#### PURE SIM
# define full joint PDF
o_EXP = tnp.zeros((NbObs_EXP+NbPred, NbObs_EXP))
o_diag=tf.concat((tf.ones(NbObs_EXP, dtype=tf.float64)),axis=0)
o_EXP = tf.linalg.set_diag(o_EXP,o_diag)

gp_model_F = tfd.JointDistributionSequentialAutoBatched( [
    make_prior_theta,
    lambda theta: tfd.MultivariateNormalDiag(
        loc=( tnp.squeeze(tnp.dot(f_sim(X_EXP,theta),o_EXP)) ),
        scale_diag=tnp.sqrt(observation_noise_variance)*tnp.ones(NbObs_EXP),
        name="observations")], validate_args=True )

# Condition full joint posterior on observations
gp_model_conditioned_F = gp_model_F.experimental_pin(observations=YObs_EXP)
