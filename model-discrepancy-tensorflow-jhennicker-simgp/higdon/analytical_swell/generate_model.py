import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.experimental.numpy as tnp

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#tf.config.run_functions_eagerly(True) # needed for numpy compatibility in tf.functions. Otherwise BUG in make_GP_SIM when avluating theta, which becomes of class 'tensorflow.python.framework.ops.Tensor' not class 'tensorflow.python.framework.ops.EagerTensor' during MCMC sampling.

tfb = tfp.bijectors
tfd = tfp.distributions
psd_kernels = tfp.math.psd_kernels

####################
# DATA PREPARATION #
####################

NbDimX = 1
NbDimY = 1
NbDimT = 2

observation_noise_variance = 0.01
observation_noise_variance_SIM = observation_noise_variance*1e-0

############ EXPERIMENTAL MODEL - DATA ############

# INPUT SPACE (OBS, PRED)
#NbObs_EXP = 6
NbPred = 10 # serves also for nb of sim. predictions

# generate Xobs for exp. observations (test points must be of shape (NbObs_EXP, NbDimX))
XObs_EXP = tnp.array((2.0000,
                      2.6000,
                      3.3800,
                      4.3940,
                      5.7120,
                      7.4260))[..., tnp.newaxis]
                      #9.6540,
                      #12.5500,
                      #16.3150,
                      #21.2090))[..., tnp.newaxis]

NbObs_EXP = len(XObs_EXP)

XPred_EXP = tnp.linspace(
    min(XObs_EXP)[0]*0.5, max(XObs_EXP)[0]*1.2, num=NbPred)[..., tnp.newaxis]
X_EXP = tnp.concatenate(
    [XObs_EXP, XPred_EXP], axis=0)

sliceObs_EXP = slice(0, NbObs_EXP)
slicePred_EXP = slice(NbObs_EXP, NbObs_EXP + NbPred)

R_Die = 1.
YObs_EXP = tnp.array((2.7587,
                      2.8780,
                      2.9404,
                      2.9918,
                      3.0718,
                      3.1419)) / (2.*R_Die)
                      #3.0958,
                      #3.0669,
                      #3.0660,
                      #3.1023)) / (2.*R_Die)

############ SIMULATION MODEL - DATA ############

# SIMULATION MODEL INPUT SPACE (OBS, PRED)
NbObs_per_dim_SIM = 2
NbObs_SIM = NbObs_per_dim_SIM ** (NbDimX + NbDimT)

# initialize Xobs0 for sim. extended exp. observations ('training?' points must be of shape (NbObs_EXP, (NbDimX + NbDimT)))
# the theta entries will change during execution (call to make_GP_SIM) and have to be initialized by zero
xx = tnp.zeros((NbObs_EXP,NbDimT), dtype=tnp.float64)
XObs0_SIM = tnp.concatenate([XObs_EXP, xx], axis=1)

# generate Xobs for sim. observations (training points must be of shape (NbObs_SIM, (NbDimX + NbDimT)))
a_obs = tnp.array([min(XObs_EXP)[0], .1, .05], dtype=tnp.float64)  # [shearRate, llambda, beta] (min)
b_obs = tnp.array([max(XObs_EXP)[0], 1., .3], dtype=tnp.float64)  # [shearRate, llambda, beta] (max)
xx = tf.transpose( tnp.linspace(a_obs, b_obs, NbObs_per_dim_SIM) )
xxx = tnp.meshgrid(*xx)
XObs_SIM = tf.transpose( tf.reshape( tnp.array(xxx), (len(a_obs),-1) ) )

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
def f_sim(x, t):
    shearRate = x[...,0]
    llambda = tf.reshape( t[...,0], -1)
    beta = tf.reshape( t[...,1], -1)
    #llambda = t[..., tnp.newaxis, 0]
    #beta = t[..., tnp.newaxis, 1]
    Sw = (1-beta)*llambda*shearRate
    return (1+Sw**2/2)**(1/6) + .13

YObs_SIM = f_sim(XObs_SIM[..., 0:NbDimX], XObs_SIM[..., NbDimX:NbDimX+NbDimT])

############ JOINT MODEL - DATA ############

ZObs = tnp.concatenate([YObs_EXP,YObs_SIM])


##########################
# PORBABILISTIC MODELING #
##########################


############ EXPERIMENTAL MODEL - MD GP DISTRIBUTION ############

def make_GP_MD(length_scale_MD, amplitude_MD, loc=None):
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

    gp = tfp.distributions.MultivariateNormalLinearOperator(
        loc=loc,
        scale=scale,
        validate_args=True,
        allow_nan_stats=False,
        name="GP_MD")

    return gp

############ SIMULATION MODEL - SIM GP DISTRIBUTION ############

def make_GP_SIM(theta, scale_diag_SIM, amplitude_SIM, loc=None):#, X_SIM=X_SIM):    
    # W: ensure that amplitude.dtype = scale_diag = theta.dtype
    # NOTE: loc == None implies zero mean.
    jitter = 1E-5

    # TODO: Implement manually.
    kernel = psd_kernels.ExponentiatedQuadratic(amplitude_SIM, length_scale=1.) #tf.ones_like(amplitude_SIM))

    kernel = tfp.math.psd_kernels.FeatureScaled(kernel, scale_diag_SIM)

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

def make_prior_amplitude_MD():
    return tfd.LogNormal(tnp.float64(0.0), tnp.float64(1.0), name="amplitude_MD")

def make_prior_length_scale_MD():
    return tfd.LogNormal(tnp.float64(0.0), tnp.float64(1.0), name="length_scale_MD")

def make_prior_amplitude_SIM():
    return tfd.LogNormal(tnp.float64(0.), tnp.float64(1.), name="amplitude_SIM")

def make_prior_length_scale_SIM():
    return tfd.Sample(tfd.LogNormal(tnp.float64(0.), tnp.float64(1.)), sample_shape=NbDimX+NbDimT, name="scale_diag_SIM")

def make_prior_theta():
    #tfd.MultivariateNormalDiag(loc=tnp.array([1.2,.2],dtype=tnp.float64), scale_diag=tnp.array([.2,.2],dtype=tnp.float64), name="theta"), # theta = (llambda, beta)
    #tfd.Sample(tfd.Uniform(tnp.float64(.1), tnp.float64(1.)), sample_shape=NbDimT, name="theta"), # theta = (llambda, beta) ## NOTE: shape is two dimensional (1,:) in order to allow for broadcasting with shape (..., NbXsim, :) in make_GP_SIM
    pdf = tfd. Independent(
        tfb.Chain(
            [ tfb.Shift(shift=tnp.array((0.,0.))), tfb.Scale(scale=tnp.array((2.,.5))) ]
        ) ( tfd.Beta(tnp.array((2.,2.)),tnp.array((2.,2.))) ),
        reinterpreted_batch_ndims=1, name="theta") # theta = (llambda, beta)
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
gp_model = tfd.JointDistributionSequentialAutoBatched( [
    make_prior_amplitude_MD,
    make_prior_length_scale_MD,
    make_GP_MD,
    #lambda length_scale_MD, amplitude_MD
    make_prior_amplitude_SIM,
    make_prior_length_scale_SIM,
    make_prior_theta,
    make_GP_SIM,
    lambda GP_SIM, theta, scale_diag_SIM, amplitude_SIM, GP_MD: tfd.MultivariateNormalDiag(
        #loc=( tf.squeeze((o_shape_EXP*GP_MD)@O_EXP + (o_shape_SIM*GP_SIM)@O_SIM )) # works in MCMC with nchain=1
        #loc=( tf.reshape( (o_shape_EXP*GP_MD)@O_EXP + (o_shape_SIM*GP_SIM)@O_SIM, -1 ) )
        #loc=( tf.matmul(o_shape_EXP*GP_MD,O_EXP)+tf.matmul(o_shape_SIM*GP_SIM,O_SIM) )
        loc=( tnp.dot(GP_MD,O_EXP)+tnp.dot(GP_SIM,O_SIM) ) # works in MCMC with nchain=1
        , scale_diag=observation_noise_diag_std, name="observations")
] )

# Condition full joint posterior on observations
gp_model_conditioned = gp_model.experimental_pin(observations=ZObs)


