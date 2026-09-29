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

# MCMC parameters
num_results = 100
num_burnin_steps = int(num_results*0.1)

####################
# DATA PREPARATION #
####################

NbDimX = 1
NbDimY = 1
NbDimT = 1

############ EXPERIMENTAL MODEL - DATA ############

# INPUT SPACE (OBS, PRED)
NbObs_EXP = 12
NbPred = 40 # serves also for nb of sim. predictions
observation_noise_variance = 0.01 *.1

# generate Xobs for exp. observations (test points must be of shape (NbObs_EXP, NbDimX))
XObs_EXP = tnp.linspace(
    1.5, 4.5, num=NbObs_EXP)[..., tnp.newaxis]
XPred_EXP = tnp.linspace(
    0., 6.5, num=NbPred)[..., tnp.newaxis]
X_EXP = tnp.concatenate(
    [XObs_EXP, XPred_EXP], axis=0)

sliceObs_EXP = slice(0, NbObs_EXP)
slicePred_EXP = slice(NbObs_EXP, NbObs_EXP + NbPred)

# synthetic experiments
# example form paper [Brynjarsdottir & O'Hagen., 2014]
def f_exp(x):
    X = x[...,0]
    beta = .25
    a = 3.
    return beta*X/(1+X/a)

YObs_EXP = tf.random.normal(
    shape=(NbObs_EXP,),
    mean=f_exp(XObs_EXP[..., 0:NbDimX]),
    stddev=tnp.sqrt(observation_noise_variance),
    seed=42,
    dtype=tf.float64)

############ SIMULATION MODEL - DATA ############

# simulator
def f_sim(x, t):
    X = x[...,0]
    beta = t[...,tnp.newaxis,0] #*0.
    return beta*X

#YObs_SIM = f_sim(XObs_SIM[..., 0:NbDimX], XObs_SIM[..., NbDimX:NbDimX+NbDimT])

############ JOINT MODEL - DATA ############

ZObs = YObs_EXP


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


############ JOINT MODEL - (FULL JOINT) POSTERIOR DISTRIBUTION ############
O_EXP = tnp.zeros((NbObs_EXP+NbPred, NbObs_EXP))
diag=tf.ones(NbObs_EXP, dtype=tf.float64)
O_EXP = tf.linalg.set_diag(O_EXP,diag)
#O_EXP[indDiag_EXP, indDiag_EXP] = 1

# define full joint PDF
gp_model = tfd.JointDistributionSequentialAutoBatched( [
    tfd.LogNormal(tnp.float64(0.0), tnp.float64(2.0), name="amplitude_MD"),
    tfd.LogNormal(tnp.float64(0.0), tnp.float64(2.0), name="length_scale_MD"),
    make_GP_MD,
    tfd.Sample(tfd.Uniform(tnp.float64(0.01), tnp.float64(2)), sample_shape=NbDimT, name="theta"), # theta = (llambda, beta)
    lambda theta, GP_MD: tfd.MultivariateNormalDiag(
        #loc=( tf.squeeze((o_shape_EXP*GP_MD)@O_EXP + (o_shape_SIM*GP_SIM)@O_SIM )) # works in MCMC with nchain=1
        #loc=( tf.reshape( (o_shape_EXP*GP_MD)@O_EXP + (o_shape_SIM*GP_SIM)@O_SIM, -1 ) )
        #loc=( tf.matmul(o_shape_EXP*GP_MD,O_EXP)+tf.matmul(o_shape_SIM*GP_SIM,O_SIM) )
        loc=( tnp.dot(GP_MD,O_EXP) + f_sim(XObs_EXP, theta) ), scale_identity_multiplier=tnp.sqrt(observation_noise_variance), name="observations")
        #loc=( f_sim(XObs_EXP, theta) ), scale_identity_multiplier=tnp.sqrt(observation_noise_variance), name="observations") # pure SIM (no MD)
] )

# Condition full joint posterior on observations
gp_model_conditioned = gp_model.experimental_pin(observations=ZObs)

############ MCMC SAMPLING ############
@tf.function(autograph=False, jit_compile=True)
def do_sampling(model, nchains, num_results, num_burnin_steps):
    sample = gp_model_conditioned.sample_unpinned(nchains)
    prob = gp_model_conditioned.unnormalized_log_prob(sample)
    constraining_bijector = gp_model_conditioned.experimental_default_event_space_bijector()
    # NOTE: Can extend left-most dimension to run multiple chains but made runtime slower:
    x_init = .5
    current_state = [x_init*tf.ones_like(sample_component) for sample_component in sample]
    scale = 0.001
    step_size = [scale*tf.ones_like(sample_component) for sample_component in sample]
    #
    nuts = tfp.mcmc.NoUTurnSampler(target_log_prob_fn=gp_model_conditioned.unnormalized_log_prob,
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
theta_samples = samples[3]

# Output chain to file
try:
    os.mkdir("output")
except:
    pass

np.savez("output/data_config.npz", YObs_EXP.numpy(), tf.reshape(X_EXP,-1).numpy(), sliceObs_EXP, slicePred_EXP)
np.save("output/amplitude_samples_MD.npy", amplitude_samples_MD.numpy())
np.save("output/length_scale_samples_MD.npy", length_scale_samples_MD.numpy())
np.save("output/gp_samples_MD.npy", gp_samples_MD.numpy())
np.save("output/theta_samples.npy", theta_samples.numpy())
np.save("output/log_accept_ratio.npy", log_accept_ratio.numpy())
#np.save("output/generated_observations.npy", generated_observations)

p_accept = tf.math.exp(tfp.math.reduce_logmeanexp(tf.minimum(
    log_accept_ratio, 0.)))
print(f"Acceptance ratio: {p_accept}")


############ PLOTS ############

# Gaussian process MD
#_ = plt.figure()
#gp_mean_MD = tf.reshape( tnp.mean(tnp.mean(gp_samples_MD, axis=0), axis=0), -1)
#gp_variance_MD = tf.reshape( tnp.mean(tnp.var(gp_samples_MD, axis=0), axis=0), -1)
#plt.plot(X_EXP[sliceObs_EXP], YObs_EXP, 'o',  c='r')
##plt.plot(tnp.vstack((X_SIM[sliceObs_EXP],X_SIM[sliceObs_SIM]))[...,0], ZObs, 'o',  c='r')
##plt.plot(X_SIM[sliceObs_SIM,0], YObs_SIM, 'o',  c='r', alpha=0.5)
#plt.plot(X_EXP[slicePred_EXP], gp_mean_MD[slicePred_EXP], c='r')
#plt.plot(X_EXP[slicePred_EXP], (gp_mean_MD[slicePred_EXP] + 2.0 * tnp.sqrt(gp_variance_MD[slicePred_EXP])), c='r', alpha=0.15)
#plt.plot(X_EXP[slicePred_EXP], (gp_mean_MD[slicePred_EXP] - 2.0 * tnp.sqrt(gp_variance_MD[slicePred_EXP])), c='r', alpha=0.15)
#plt.savefig("output/plot_GPMD.png")
#plt.clf()

# parameters
theta_mean=tf.reshape( tnp.mean(tnp.mean(theta_samples,axis=0),axis=0), -1)
print('theta_mean =', theta_mean)
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

# Condition full joint posterior on MAP/MEAN estimates
#gp_model_PPC = gp_model.experimental_pin(theta=theta_mean, length_scale_MD=length_scale_mean_MD, amplitude_MD=amplitude_mean_MD)

GP_PPC = gp_model.experimental_pin(
    amplitude_MD=amplitude_samples_MD,
    length_scale_MD=length_scale_samples_MD,
    GP_MD=gp_samples_MD,
    theta=theta_samples)

num_results_PPC = 1000
gp_samples_PPC = GP_PPC.sample_unpinned(num_results_PPC)[-1]
np.save("output/gp_samples_PPC.npy", gp_samples_PPC.numpy())

gp_mean_PPC = tnp.mean(gp_samples_PPC, axis=range(gp_samples_PPC.ndim-1))
#gp_variance_PPC = tnp.var(gp_samples_PPC, axis=range(gp_samples_PPC.ndim-1))
gp_variance_PPC = tnp.sum(tnp.var(gp_samples_PPC, axis=0),
                          axis=range(tnp.var(gp_samples_PPC, axis=0).ndim-1)
                          )/ num_results**2
err = tnp.sqrt(tnp.sum((gp_mean_PPC-YObs_EXP)**2))
print("err PCC:", err)

# plot PPC
_ = plt.figure()
plt.plot(X_EXP[sliceObs_EXP], YObs_EXP, 'o',  c='black', label=r"training data")
plt.plot(X_EXP[sliceObs_EXP], gp_mean_PPC, 'x',  c='black', label=r"PPC")

upper_PPC = gp_mean_PPC + 2.0 * np.sqrt(gp_variance_PPC)
lower_PPC = gp_mean_PPC - 2.0 * np.sqrt(gp_variance_PPC)
plt.fill_between(X_EXP[sliceObs_EXP][:,0], lower_PPC, upper_PPC,
                 alpha=0.2, color='tab:grey', label=r"PP Credibility interval, $\pm 2 \sigma$")

plt.savefig(os.path.join("output/plot_OBS_PPC.png"))
plt.clf()

# Alternative calculus:
#gp_samples_PPC_full = gp_samples
#gp_samples_PPC = tnp.mean(gp_samples_PPC_full, axis=range(1,gp_samples_PPC_full.ndim-1))
#gp_mean_PPC = tnp.mean(gp_samples_PPC, axis=range(gp_samples_PPC.ndim-1))
#gp_variance_PPC = tnp.sum(tnp.var(gp_samples_PPC_full, axis=0),axis=(0,1)) / num_results**2

# Another alternative: not fixing Y_MD=realiz(GP_MD(X; amplitude_samples_MD,length_scale_MD)) in the PP distribution
gp_ppc=gp_model.experimental_pin(amplitude_MD=amplitude_samples_MD,length_scale_MD=length_scale_samples_MD,theta=theta_samples)

# plot approach 0
s0=GP_PPC.sample_unpinned(num_results_PPC)[-1]
m0=tnp.mean(s0, axis=range(s0.ndim-1))
v0=tnp.var(s0, axis=range(s0.ndim-1))
#v0=tnp.sum(tnp.var(s0, axis=0),range(tnp.var(s0, axis=0).ndim-1)) / num_results**2
_ = plt.figure()
plt.plot(X_EXP[sliceObs_EXP][:,0], YObs_EXP, 'o',  c='black', label=r"training data")
plt.plot(X_EXP[sliceObs_EXP][:,0], m0, 'x',  c='black', label=r"PPC")
u0=m0+2.0 * np.sqrt(v0)
l0=m0-2.0 * np.sqrt(v0)
plt.fill_between(X_EXP[sliceObs_EXP][:,0], l0,u0,
                 alpha=0.2, color='tab:grey', label=r"PP Credibility interval, $\pm 2 \sigma$")

plt.savefig(os.path.join("output/plot_OBS_PPC0.png"))
plt.clf()
# plot approach 1
s1=gp_ppc.sample_unpinned(num_results_PPC)[-1]
m1=tnp.mean(s1, axis=range(s1.ndim-1))
#v1=tnp.var(s1, axis=range(s1.ndim-1))
v1=tnp.sum(tnp.var(s1, axis=0),range(tnp.var(s1, axis=0).ndim-1)) / num_results**2
_ = plt.figure()
plt.plot(X_EXP[sliceObs_EXP][:,0], YObs_EXP, 'o',  c='black', label=r"training data")
plt.plot(X_EXP[sliceObs_EXP][:,0], m1, 'x',  c='black', label=r"PPC")
u1=m1+2.0 * np.sqrt(v1)
l1=m1-2.0 * np.sqrt(v1)
plt.fill_between(X_EXP[sliceObs_EXP][:,0], l1,u1,
                 alpha=0.2, color='tab:grey', label=r"PP Credibility interval, $\pm 2 \sigma$")

plt.savefig(os.path.join("output/plot_OBS_PPC1.png"))
plt.clf()
# plot approach 1 with predictions at XPred
s1p=gp_ppc.sample_unpinned(num_results_PPC)[0] #+ f_sim(X_EXP, theta_samples)
m1p=tnp.mean(s1p, axis=range(s1p.ndim-1))
v1p=tnp.sum(tnp.var(s1p, axis=0),range(tnp.var(s1p, axis=0).ndim-1)) / num_results**2
_ = plt.figure()
plt.plot(X_EXP[sliceObs_EXP][:,0], YObs_EXP, 'o',  c='black', label=r"training data")
plt.plot(X_EXP[:,0], m1p, 'x',  c='black', label=r"PPC")
u1p=m1p+2.0 * np.sqrt(v1p)
l1p=m1p-2.0 * np.sqrt(v1p)
plt.fill_between(X_EXP[:,0], l1p,u1p,
                 alpha=0.2, color='tab:grey', label=r"PP Credibility interval, $\pm 2 \sigma$")

plt.savefig(os.path.join("output/plot_OBS_PPC1p.png"))
plt.clf()
