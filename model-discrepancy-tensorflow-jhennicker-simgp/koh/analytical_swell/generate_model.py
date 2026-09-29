import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf
import tensorflow.experimental.numpy as tnp

#tf.config.experimental.enable_tensor_float_32_execution(False)

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

############ EXPERIMENTAL MODEL - DATA ############

# INPUT SPACE (OBS, PRED)
#NbObs_EXP = 10
NbPred = 10 # serves also for nb of sim. predictions
observation_noise_variance = 0.01

# generate Xobs for exp. observations (test points must be of shape (NbObs_EXP, NbDimX))

# SBR data 0
XObs_EXP = tnp.array((2.0000,2.6000,3.3800,4.3940,5.7120,7.4260))[..., tnp.newaxis]
                      #,9.6540,12.5500,16.3150,21.2090))[..., tnp.newaxis]
                      #,21.2090))[..., tnp.newaxis]
# polyethylene data
#XObs_EXP = tnp.array((2.0000,2.6060,3.3960,4.4240,5.7650,7.5120,9.7880,12.7540))[..., tnp.newaxis]
#16.6180,21.6540,28.2150,36.7639,47.9030,62.4181,81.3299,105.9730,138.0830,179.9221,234.4392,305.4744

NbObs_EXP = len(XObs_EXP)

XPred_EXP = tnp.linspace(
    min(XObs_EXP)[0]*0.5, max(XObs_EXP)[0]*1.2, num=NbPred)[..., tnp.newaxis]
X_EXP = tnp.concatenate(
    [XObs_EXP, XPred_EXP], axis=0)

sliceObs_EXP = slice(0, NbObs_EXP)
slicePred_EXP = slice(NbObs_EXP, NbObs_EXP + NbPred)

# SBR data 0
R_Die = 1.
YObs_EXP = tnp.array((2.7587,2.8780,2.9404,2.9918,3.0718,3.1419)) / (2.*R_Die)
                      #3.0958,3.0669,3.0660,3.1023)) / (2.*R_Die)
                      #,3.1023)) / (2.*R_Die)
# polyethylene data
#YObs_EXP = tnp.array((1.3115,1.2911,1.3041,1.3785,1.3537,1.3903,1.3842,1.4109))
#1.4090,1.4537,1.4521,1.4881,1.4792,1.4702,1.4866,1.5242,1.6083,1.6038,1.7944,1.7538

############ SIMULATION MODEL - DATA ############

# simulator
def f_sim(x, t):
    shearRate = x[...,0]
    llambda = t[..., tnp.newaxis, 0]
    beta = t[..., tnp.newaxis, 1]
    Sw = (1-beta)*llambda*shearRate # recoverable shear (at the die wall)
    return ( (1+Sw**2/2)**(1/6) + .13 )

#YObs_SIM = f_sim(XObs_SIM[..., 0:NbDimX], XObs_SIM[..., NbDimX:NbDimX+NbDimT])

############ JOINT MODEL - DATA ############

ZObs = YObs_EXP


##########################
# PORBABILISTIC MODELING #
##########################

############ EXPERIMENTAL MODEL - MD GP DISTRIBUTION ############

def make_GP_MD(mu, loc=None):
    # NOTE: loc == None implies zero mean.
    jitter = 1E-5
    # hyperparameters mu
    amplitude_MD = mu[...,0]
    length_scale_MD = mu[...,1]

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
        allow_nan_stats=False)

    return gp


############ JOINT MODEL - (FULL JOINT) POSTERIOR DISTRIBUTION ############
O_EXP = tnp.zeros((NbObs_EXP+NbPred, NbObs_EXP))
diag=tf.ones(NbObs_EXP, dtype=tf.float64)
O_EXP = tf.linalg.set_diag(O_EXP,diag)
#O_EXP[indDiag_EXP, indDiag_EXP] = 1

# define full joint PDF
gp_model = tfd.JointDistributionSequentialAutoBatched( [
    #tfd.LogNormal(tnp.float64(0.), tnp.float64(1.), name="amplitude_MD"),
    #tfd.LogNormal(tnp.float64(1.), tnp.float64(1.), name="length_scale_MD"),
    tfd. Independent(tfd.LogNormal(loc=tnp.array((0.,1.)), scale=tnp.array((1.,1.))), reinterpreted_batch_ndims=1, name="mu"), # mu = (amplitude_MD, length_scale_MD)
    make_GP_MD,
    #tfd.Sample(tfd.Uniform(tnp.float64(0.), tnp.float64(2.)), sample_shape=NbDimT, name="theta"), # theta = (llambda, beta)
    #tfd.MultivariateNormalDiag(loc=tnp.array((0.4,0.1),dtype=tnp.float64), scale_diag=tnp.array((1.,1.),dtype=tnp.float64), name="theta"), # theta = (llambda, beta)
    tfd. Independent(
        tfb.Chain(
            [ tfb.Shift(shift=tnp.array((0.,0.))), tfb.Scale(scale=tnp.array((2.,.5))) ]
        ) ( tfd.Beta(tnp.array((2.,2.)),tnp.array((2.,2.))) ),
        reinterpreted_batch_ndims=1, name="theta"), # theta = (llambda, beta)
    lambda theta, GP_MD: tfd.MultivariateNormalDiag(
        #loc=( tf.squeeze((o_shape_EXP*GP_MD)@O_EXP + (o_shape_SIM*GP_SIM)@O_SIM )) # works in MCMC with nchain=1
        #loc=( tf.reshape( (o_shape_EXP*GP_MD)@O_EXP + (o_shape_SIM*GP_SIM)@O_SIM, -1 ) )
        #loc=( tf.matmul(o_shape_EXP*GP_MD,O_EXP)+tf.matmul(o_shape_SIM*GP_SIM,O_SIM) )
        loc=( tnp.dot(GP_MD,O_EXP) + f_sim(XObs_EXP, theta) ), scale_identity_multiplier=tnp.sqrt(observation_noise_variance), name="observations")
] )

# Condition full joint posterior on observations
gp_model_conditioned = gp_model.experimental_pin(observations=ZObs)
