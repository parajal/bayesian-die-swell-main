import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.experimental.numpy as tnp

#tf.config.experimental.enable_tensor_float_32_execution(False)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#tf.config.run_functions_eagerly(True) # needed for numpy compatibility in tf.functions. Otherwise BUG in make_GP_SIM when avluating theta, which becomes of class 'tensorflow.python.framework.ops.Tensor' not class 'tensorflow.python.framework.ops.EagerTensor' during MCMC sampling.

tfb = tfp.bijectors
tfd = tfp.distributions
psd_kernels = tfp.math.psd_kernels

# MCMC parameters
num_results = 1000
num_burnin_steps = int(num_results*0.1)

####################
# DATA PREPARATION #
####################

NbDimX = 1
NbDimY = 1
NbDimT = 1

############ EXPERIMENTAL MODEL - DATA ############

# INPUT SPACE (OBS, PRED)
NbObs_EXP = 10
NbPred = 10 # serves also for nb of sim. predictions
observation_noise_variance = 0.01

# generate Xobs for exp. observations (test points must be of shape (NbObs_EXP, NbDimX))
XObs_EXP = tnp.linspace(
    0.0, 2.0*tnp.pi, num=NbObs_EXP)[..., tnp.newaxis]
XPred_EXP = tnp.linspace(
    -0.5, 2.0*tnp.pi + 0.5, num=NbPred)[..., tnp.newaxis]
X_EXP = tnp.concatenate(
    [XObs_EXP, XPred_EXP], axis=0)

sliceObs_EXP = slice(0, NbObs_EXP)
slicePred_EXP = slice(NbObs_EXP, NbObs_EXP + NbPred)

# synthetic experiments
def f_exp(x):
    X = x[...,0]
    beta = 0.25
    eps = 1.
    return beta*X + eps*tnp.sin(X)

YObs_EXP = tf.random.normal(
    shape=(NbObs_EXP,),
    mean=f_exp(XObs_EXP[..., 0:NbDimX]),
    stddev=tnp.sqrt(observation_noise_variance),
    dtype=tf.float64)

############ SIMULATION MODEL - DATA ############

# SIMULATION MODEL INPUT SPACE (OBS, PRED)
NbObs_per_dim_SIM = 4
NbObs_SIM = NbObs_per_dim_SIM ** (NbDimX + NbDimT)

# initialize Xobs0 for sim. extended exp. observations (test points must be of shape (NbObs_EXP, (NbDimX + NbDimT)))
# the theta entries will change during execution (call to make_GP_SIM) and have to be initialized by zero
xx = tnp.zeros((NbObs_EXP,NbDimT), dtype=tnp.float64)
XObs0_SIM = tnp.concatenate([XObs_EXP, xx], axis=1)

# generate Xobs for sim. observations (training points must be of shape (NbObs_SIM, (NbDimX + NbDimT)))
a_obs = tnp.array([.1, 0.1], dtype=tnp.float64)  # [shearRate, llambda, beta] (min)
b_obs = tnp.array([2.0*tnp.pi, 1.], dtype=tnp.float64)  # [shearRate, llambda, beta] (max)
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
    X = x[...,0]
    beta = t[...,tnp.newaxis,0]
    return beta*X

YObs_SIM = tf.reshape(
    tnp.array([f_sim(XObs_SIM[j, 0:NbDimX], XObs_SIM[j, NbDimX:NbDimX+NbDimT]) for j in range(NbObs_SIM)]), -1)

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


############ JOINT MODEL - (FULL JOINT) POSTERIOR DISTRIBUTION ############
# define projection tensors for the likelihood data GP_MD and GP_SIM: setting prediction output to zero AND inclusion mapping into the correct observation space ~ {ZObs}
O_EXP = tnp.zeros((NbObs_EXP+NbPred, NbObs_EXP+NbObs_SIM))
diag=tf.concat((tf.ones(NbObs_EXP, dtype=tf.float64),tf.zeros(min(NbPred,NbObs_SIM), dtype=tf.float64)),axis=0)
O_EXP = tf.linalg.set_diag(O_EXP,diag)
#O_EXP[indDiag_EXP, indDiag_EXP] = 1
O_SIM = tnp.zeros((NbObs_EXP+NbObs_SIM+NbPred, NbObs_EXP+NbObs_SIM))
diag=tf.ones(NbObs_EXP+NbObs_SIM, dtype=tf.float64)
O_SIM = tf.linalg.set_diag(O_SIM,diag)
#O_SIM[indDiag_SIM, indDiag_SIM] = 1

# define full joint PDF
gp_model = tfd.JointDistributionSequentialAutoBatched( [
    tfd.LogNormal(tnp.float64(0.0), tnp.float64(2.0), name="amplitude_MD"),
    tfd.LogNormal(tnp.float64(0.0), tnp.float64(2.0), name="length_scale_MD"),
    make_GP_MD,
    tfd.LogNormal(tnp.float64(0), tnp.float64(1), name="amplitude_SIM"),
    tfd.Sample(tfd.LogNormal(tnp.float64(0), tnp.float64(1)), sample_shape=NbDimX+NbDimT, name="scale_diag_SIM"),
    tfd.Sample(tfd.Uniform(tnp.float64(0), tnp.float64(2)), sample_shape=NbDimT, name="theta"), # theta = (llambda, beta)
    make_GP_SIM,
    lambda GP_SIM, theta, scale_diag_SIM, amplitude_SIM, GP_MD:
    tfd.MultivariateNormalDiag(
        loc=( tnp.dot(GP_MD,O_EXP) + tnp.dot(GP_SIM,O_SIM) ), scale_identity_multiplier=tnp.sqrt(observation_noise_variance), name="observations")
] )

# Condition full joint posterior on observations
gp_model_conditioned = gp_model.experimental_pin(observations=ZObs)

############ MCMC SAMPLING ############
@tf.function(autograph=False, jit_compile=True)
def do_sampling(model, nchains, num_results, num_burnin_steps):
    try:
        sample = model.sample(nchains)
    except AttributeError:
        sample = model.sample_unpinned(nchains)
    except:
        raise
    prob = model.unnormalized_log_prob(sample)
    constraining_bijector = model.experimental_default_event_space_bijector()
    # NOTE: Can extend left-most dimension to run multiple chains but made runtime slower:
    #x_init = 1.
    #current_state = [x_init*tf.ones_like(sample_component) for sample_component in sample]
    current_state = sample
    scale = 0.01
    step_size = [scale*tf.ones_like(sample_component) for sample_component in sample]
    #s=model.sample_unpinned((100,nchains))
    #sample = [tnp.mean(sample_component,axis=0) for sample_component in s]
    #prob = model.unnormalized_log_prob(sample)
    #constraining_bijector = model.experimental_default_event_space_bijector()
    #current_state = sample
    #scale = 0.001
    #step_size = [scale*tnp.var(sample_component,axis=0) for sample_component in s]
    #
    nuts = tfp.mcmc.NoUTurnSampler(target_log_prob_fn=model.unnormalized_log_prob,
                                   step_size=step_size)
    transformed_nuts = tfp.mcmc.TransformedTransitionKernel(
        nuts, bijector=constraining_bijector)
    adapted_transformed_nuts = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=transformed_nuts, num_adaptation_steps=int(num_burnin_steps * 0.8))

    return tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=current_state,
        kernel=adapted_transformed_nuts,
        num_burnin_steps=num_burnin_steps,
        num_steps_between_results=0,
        trace_fn=lambda _, pkr: [pkr.inner_results.inner_results.step_size, pkr.inner_results.inner_results.log_accept_ratio])

print("Begin chain...")
nchains=1
samples, [step_size, log_accept_ratio] = do_sampling(model=gp_model_conditioned, nchains=nchains, num_results=num_results, num_burnin_steps=num_burnin_steps)
print("Chain finished.")

############ SAVING OUTPUT TO FILES ############

amplitude_samples_MD = samples[0]
length_scale_samples_MD = samples[1]
gp_samples_MD = samples[2]
amplitude_samples_SIM = samples[3]
length_scale_samples_SIM = samples[4]
theta_samples = samples[5]
gp_samples_SIM = samples[6]

# Output chain to file
try:
    os.mkdir("output")
except:
    pass

np.savez("output/data_config_EXP.npz", YObs_EXP.numpy(), tf.reshape(X_EXP,-1).numpy(), sliceObs_EXP, slicePred_EXP)
np.savez("output/data_config_SIM.npz", YObs_SIM.numpy(), X_SIM.numpy(), sliceObs_SIM, slicePred_SIM)
np.save("output/amplitude_samples_MD.npy", amplitude_samples_MD.numpy())
np.save("output/length_scale_samples_MD.npy", length_scale_samples_MD.numpy())
np.save("output/gp_samples_MD.npy", gp_samples_MD.numpy())
np.save("output/amplitude_samples_SIM.npy", amplitude_samples_SIM.numpy())
np.save("output/length_scale_samples_SIM.npy", length_scale_samples_SIM.numpy())
np.save("output/theta_samples.npy", theta_samples.numpy())
np.save("output/gp_samples_SIM.npy", gp_samples_SIM.numpy())
np.save("output/log_accept_ratio.npy", log_accept_ratio.numpy())
#np.save("output/generated_observations.npy", generated_observations)

p_accept = tf.math.exp(tfp.math.reduce_logmeanexp(tf.minimum(
    log_accept_ratio, 0.)))
print(f"Acceptance ratio: {p_accept}")


############ PLOTS ############

# Gaussian process MD
_ = plt.figure()
gp_mean_MD = tf.reshape( tnp.mean(tnp.mean(gp_samples_MD, axis=0), axis=0), -1)
gp_variance_MD = tf.reshape( tnp.mean(tnp.var(gp_samples_MD, axis=0), axis=0), -1)
plt.plot(X_EXP[sliceObs_EXP], YObs_EXP, 'o',  c='r')
#plt.plot(tnp.vstack((X_SIM[sliceObs_EXP],X_SIM[sliceObs_SIM]))[...,0], ZObs, 'o',  c='r')
#plt.plot(X_SIM[sliceObs_SIM,0], YObs_SIM, 'o',  c='r', alpha=0.5)
plt.plot(X_EXP[slicePred_EXP], gp_mean_MD[slicePred_EXP], c='r')
plt.plot(X_EXP[slicePred_EXP], (gp_mean_MD[slicePred_EXP] + 2.0 * tnp.sqrt(gp_variance_MD[slicePred_EXP])), c='r', alpha=0.15)
plt.plot(X_EXP[slicePred_EXP], (gp_mean_MD[slicePred_EXP] - 2.0 * tnp.sqrt(gp_variance_MD[slicePred_EXP])), c='r', alpha=0.15)
plt.savefig("output/plot_GPMD.png")
plt.clf()

# Gaussian process SIM
_ = plt.figure()
gp_mean_SIM = tf.reshape( tnp.mean(tnp.mean(gp_samples_SIM, axis=0), axis=0), -1)
gp_variance_SIM = tf.reshape( tnp.mean(tnp.var(gp_samples_SIM, axis=0), axis=0), -1)
plt.plot(X_SIM[sliceObs_EXP,0], YObs_EXP, 'o',  c='r')
plt.plot(X_SIM[sliceObs_SIM,0], YObs_SIM, 'o',  c='r', alpha=0.5)
plt.plot(X_SIM[slicePred_SIM,0], gp_mean_SIM[slicePred_SIM], '-', c='r')
plt.plot(X_SIM[slicePred_SIM,0], (gp_mean_SIM[slicePred_SIM] + 2.0 * tnp.sqrt(gp_variance_SIM[slicePred_SIM])), '-', c='r', alpha=0.15)
plt.plot(X_SIM[slicePred_SIM,0], (gp_mean_SIM[slicePred_SIM] - 2.0 * tnp.sqrt(gp_variance_SIM[slicePred_SIM])), '-', c='r', alpha=0.15)
plt.savefig("output/plot_GPSIM.png")
plt.clf()

# Gaussian process SIM
_ = plt.figure()
gp_mean_SIM = tf.reshape( tnp.mean(tnp.mean(gp_samples_SIM, axis=0), axis=0), -1)
gp_variance_SIM = tf.reshape( tnp.mean(tnp.var(gp_samples_SIM, axis=0), axis=0), -1)
plt.plot(X_SIM[sliceObs_EXP,0], YObs_EXP, 'o',  c='r')
plt.plot(X_SIM[sliceObs_SIM,0], YObs_SIM, 'o',  c='r', alpha=0.5)
plt.plot(X_SIM[slicePred_SIM,0], gp_mean_MD[slicePred_EXP]+gp_mean_SIM[slicePred_SIM], '-', c='r')
plt.savefig("output/plot_GPMAP.png")
plt.clf()

# parameters
theta_mean=tf.reshape( tnp.mean(tnp.mean(theta_samples,axis=0),axis=0), -1)
print('theta_mean =', theta_mean)
amplitude_mean_SIM=tf.reshape( tnp.mean(tnp.mean(amplitude_samples_SIM,axis=0),axis=0), -1)
print('amplitude_mean_SIM =', amplitude_mean_SIM)
length_scale_mean_SIM=tf.reshape( tnp.mean(tnp.mean(length_scale_samples_SIM,axis=0),axis=0), -1)
print('length_scale_mean_SIM =', length_scale_mean_SIM)
amplitude_mean_MD=tf.reshape( tnp.mean(tnp.mean(amplitude_samples_MD,axis=0),axis=0), -1)
print('amplitude_mean_MD =', amplitude_mean_MD)
length_scale_mean_MD=tf.reshape( tnp.mean(tnp.mean(length_scale_samples_MD,axis=0),axis=0), -1)
print('length_scale_mean_MD =', length_scale_mean_MD)

# PREDICTION MODEL
#gpPred_SIM = make_GP_SIM(theta_mean, length_scale_mean_SIM, amplitude_mean_SIM, loc=gp_mean_SIM)
#gpPred_MD = make_GP_MD(length_scale_mean_MD, amplitude_mean_MD, loc=gp_mean_MD)
#gp_pred = tfd.JointDistributionSequential([gp_pred_SIM, gp_pred_MD])
#print("Begin prediction chain...")
#nchains=2
#gp_pred_samples, [gp_pred_step_size, gp_pred_log_accept_ratio] = do_sampling(model=gp_pred, nchains=nchains, num_results=num_results, num_burnin_steps=num_burnin_steps)
#print("Prediction chain finished.")

GP_PPC = gp_model.experimental_pin(
    amplitude_MD=amplitude_samples_MD,
    length_scale_MD=length_scale_samples_MD,
    GP_MD=gp_samples_MD,
    amplitude_SIM=amplitude_samples_SIM,
    scale_diag_SIM=length_scale_samples_SIM,
    theta=theta_samples,
    GP_SIM=gp_samples_SIM)

nsamples_per_mcmc_sample = 1
gp_samples_PPC = GP_PPC.sample_unpinned(nsamples_per_mcmc_sample)[0]
np.save("output/gp_samples_PPC.npy", gp_samples_PPC.numpy())

gp_mean_PPC = tnp.mean(gp_samples_PPC, axis=range(gp_samples_PPC.ndim-1))
gp_variance_PPC = tnp.var(gp_samples_PPC, axis=range(gp_samples_PPC.ndim-1))
err = tnp.sqrt(tnp.sum((gp_mean_PPC-ZObs)**2))
print("err PCC:", err)

# plot PPC
# on YObs_EXP
_ = plt.figure()
plt.plot(XObs_EXP, YObs_EXP, 'o',  c='black', label=r"training data")
plt.plot(XObs_EXP[...,0], gp_mean_PPC[sliceObs_EXP], 'x',  c='black', label=r"PPC")
err_PPC = 2.0 * np.sqrt(gp_variance_PPC[sliceObs_EXP])
plt.errorbar(XObs_EXP[...,0], gp_mean_PPC[sliceObs_EXP], yerr=err_PPC, fmt='x', c='black', label=r"PPC error")
plt.legend()
plt.savefig("output/plot_PPC_EXP.png")
plt.clf()
# on YObs_SIM
_ = plt.figure()
plt.plot(XObs_SIM[...,0], YObs_SIM, 'o',  c='black', label=r"training data")
plt.plot(XObs_SIM[...,0], gp_mean_PPC[sliceObs_SIM], 'x',  c='black', label=r"PPC")
err_PPC = 2.0 * np.sqrt(gp_variance_PPC[sliceObs_SIM])
plt.errorbar(XObs_SIM[...,0], gp_mean_PPC[sliceObs_SIM], yerr=err_PPC, fmt='x', c='black', label=r"PPC error")
plt.legend()
plt.savefig("output/plot_PPC_SIM.png")
plt.clf()


# RUN MCMC ON PRIOR AND SAVE CHAIN
ind_mcmc_prior = 0
if ind_mcmc_prior == 1:
    print("Begin chain on prior PDF...")
    nchains=1
    samples_prior, [step_size_prior, log_accept_ratio_prior] = do_sampling(model=gp_model, nchains=nchains, num_results=num_results, num_burnin_steps=num_burnin_steps)
    print("Chain finished on prior PDF.")
    print("Acceptance ratio:", tf.math.exp(tfp.math.reduce_logmeanexp(tf.minimum(log_accept_ratio_prior, 0.))))
    np.save("output/log_accept_ratio_prior.npy", log_accept_ratio_prior.numpy())

else:

    samples_prior = gp_model.sample(num_results)
    
np.save("output/amplitude_samples_MD_prior.npy", samples_prior[0].numpy())
np.save("output/length_scale_samples_MD_prior.npy", samples_prior[1].numpy())
np.save("output/gp_samples_MD_prior.npy", samples_prior[2].numpy())
np.save("output/amplitude_samples_SIM_prior.npy", samples_prior[3].numpy())
np.save("output/length_scale_samples_SIM_prior.npy", samples_prior[4].numpy())
np.save("output/theta_samples_prior.npy", samples_prior[5].numpy())
np.save("output/gp_samples_SIM_prior.npy", samples_prior[6].numpy())
