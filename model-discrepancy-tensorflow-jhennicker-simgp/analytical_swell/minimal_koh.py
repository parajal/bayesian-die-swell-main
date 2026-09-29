import tensorflow as tf
import tensorflow_probability
import tensorflow.experimental.numpy as tnp

tfd = tensorflow_probability.distributions
tfp = tensorflow_probability

####################
# DATA PREPARATION #
####################

############ EXPERIMENTAL MODEL - DATA ############

# INPUT SPACE (OBS, PRED)

x_obs = tnp.array((2.6000, 3.3800, 4.3940, 5.7120, 7.4260,
                  9.6540, 12.5500))[..., tnp.newaxis]
y_obs = tnp.array((1.4363, 1.4791, 1.4997, 1.49255, 1.55585, 1.56545, 1.59275))

x_pred = tnp.linspace(2., 16., num=10)[..., tnp.newaxis]


############ SIMULATION MODEL - DATA #

# simulator
def f_sim(x, t):
    shearRate = x[..., 0]
    llambda = t[...,0]
    beta = t[...,1]
    Sw = (1 - beta) * llambda * shearRate
    return (1 + Sw**2 / 2)**(1 / 6) + .13


##########################
# PORBABILISTIC MODELING #
##########################


############ PRIOR DEFINITIONS ############
c = tnp.float64(1.)
concentration = c
scale = c
loc = c
concentration1 = c
concentration0 = c
low = tnp.float64(0.)
high = tnp.float64(2.)


def prior_amplitude():
    return tfd.InverseGamma(concentration, scale, name="amplitude")


def prior_length_scale():
    return tfd.TruncatedNormal(loc, scale, low, high, name="length_scale")


def prior_theta():
    return tfd.Sample(
        tfd.Beta(
            concentration1,
            concentration0),
        sample_shape=2,
        name="theta")


def prior_GP(length_scale, amplitude):
    kernel = tensorflow_probability.math.psd_kernels.ExponentiatedQuadratic(
        amplitude, length_scale)
    kernel_matrix = kernel.matrix(
        tnp.concatenate(
            (x_obs, x_pred)), tnp.concatenate(
            (x_obs, x_pred)))
    kernel_matrix = tf.linalg.set_diag(
        kernel_matrix,
        tf.linalg.diag_part(kernel_matrix) +
        1E-5)  # Add jitter
    scale = tf.linalg.LinearOperatorLowerTriangular(
        tf.linalg.cholesky(kernel_matrix))
    return tfd.MultivariateNormalLinearOperator(scale=scale, name="GP")


########## LIKELIHOOD #############
noise_stddev = tnp.sqrt(0.0025)


def likelihood(theta, GP):
    loc = GP[0:len(x_obs)] + f_sim(x_obs, theta)
    return tfd.MultivariateNormalDiag(
        loc,
        scale_identity_multiplier=noise_stddev,
        name="observations")


############ JOINT MODEL ############

model = tfd.JointDistributionSequentialAutoBatched([
    prior_amplitude,
    prior_length_scale,
    prior_GP,
    prior_theta,
    likelihood
])


########  POSTERIOR DISTRIBUTION ##########

model_conditioned = model.experimental_pin(observations=y_obs)



##########################
#      MCMC SAMPLING     #
##########################
num_results = 1000
num_burnin_steps = int(num_results*0.1)
nchains=1
samples_for_prior_mean = model_conditioned.sample_unpinned((100, nchains))
step_size = [tnp.std(sample_component, axis=0) for sample_component in samples_for_prior_mean]
current_state = model_conditioned.sample_unpinned(nchains)

@tf.function(autograph=False, jit_compile=True)
def do_sampling():
    
    constraining_bijector = model_conditioned.experimental_default_event_space_bijector()
    nuts = tfp.mcmc.NoUTurnSampler(target_log_prob_fn=model_conditioned.unnormalized_log_prob,
                                   step_size=step_size)
    transformed_nuts = tfp.mcmc.TransformedTransitionKernel(
        nuts, bijector=constraining_bijector)
    adapted_transformed_nuts = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=transformed_nuts,
        num_adaptation_steps=int(num_burnin_steps * 0.8),
        target_accept_prob=0.6, # default =0.75
        validate_args=True)
    samples, pkr = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=current_state,
        kernel=adapted_transformed_nuts,
        num_burnin_steps=num_burnin_steps,
        num_steps_between_results=2,
        trace_fn=lambda _, pkr: [pkr.inner_results.inner_results.step_size, pkr.inner_results.inner_results.log_accept_ratio])

    return samples, pkr


print("Begin chain...")
#tf.config.run_functions_eagerly(True)
samples, [step_size, log_accept_ratio] = do_sampling()
print("Chain finished.")


############ SAVING OUTPUT TO FILES ############
import matplotlib.pyplot as plt

mean_GP = tnp.mean(samples[2], axis=range(2))
mean_theta = tnp.mean(samples[3], axis=range(2))
mean_fsim = tnp.mean(f_sim(x_pred, samples[3]), axis=0)
slice_pred = slice(len(x_obs),len(x_obs)+len(x_pred))

_ = plt.figure()
plt.plot(x_obs, y_obs, 's',  c='black', label=r"exp. obs.")
plt.plot(x_pred, mean_fsim, '-.', c='r', label=r"sim. model (post. mean)")
plt.plot(x_pred, mean_GP[slice_pred] + mean_fsim, '-', c='r', label=r"pred. model (post. mean)")
plt.xlabel(r"Shear Rate")
plt.ylabel(r"Die Swell")
plt.legend()
plt.savefig("plot.png")
plt.close()

