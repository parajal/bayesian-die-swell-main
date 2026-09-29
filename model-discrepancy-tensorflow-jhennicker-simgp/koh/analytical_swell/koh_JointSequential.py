import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions
psd_kernels = tfp.math.psd_kernels

num_observations = 10
num_predictions = 10
observation_noise_variance = 0.001

# MCMC parameters
num_results = 100
num_burnin_steps = int(num_results*0.1)

observation_index_points = np.linspace(4., 14., num=num_observations)[..., np.newaxis]
prediction_index_points = np.linspace(1., 20., num=num_predictions)[..., np.newaxis]
index_points = np.concatenate(
    [observation_index_points, prediction_index_points], axis=0)

observations_slice = slice(0, num_observations)
predictions_slice = slice(num_observations, num_observations + num_predictions)

# experiments
def f_exp(shearRate):
    #solvent viscosity [Pa*s] 0.638219098023993
    #polymer viscosity [Pa*s] 31918.482251172325
    #polymer relaxation time [s] 0.05869798942007365
    llambda = 0.05869798942007365 *4.
    beta = 0.638219098023993 / 31918.482251172325
    Sw = (1-beta)*llambda*shearRate
    return 0.19 + (1+Sw**2/3)**(1/4)

observations = np.random.normal(
    f_exp(observation_index_points[..., 0]), np.sqrt(observation_noise_variance))

# simulator
def f_sim(shearRate, llambda, beta):
    Sw = (1-beta)*llambda*shearRate
    return 0.19 + (1+Sw**2/3)**(1/4) #+ np.sin(shearRate)


def make_gaussian_process(amplitude, length_scale, loc=None):
    # NOTE: loc == None implies zero mean.
    jitter = 1E-5

    # TODO: Implement manually.
    kernel = psd_kernels.ExponentiatedQuadratic(amplitude, length_scale)

    kernel_matrix = kernel.matrix(
        index_points, index_points)
    # Add jitter
    kernel_matrix = tf.linalg.set_diag(
        kernel_matrix, tf.linalg.diag_part(kernel_matrix) + jitter)

    scale = tf.linalg.LinearOperatorLowerTriangular(
        tf.linalg.cholesky(kernel_matrix),
        is_non_singular=True)

    gp = tfp.distributions.mvn_linear_operator.MultivariateNormalLinearOperator(
        loc=loc,
        scale=scale,
        validate_args=True,
        allow_nan_stats=False)

    return gp


gp_model = tfd.JointDistributionSequential([tfd.LogNormal(np.float64(0.0), np.float64(2.0), name="length_scale"),
                                            tfd.LogNormal(np.float64(0.0), np.float64(2.0), name="amplitude"),
                                            make_gaussian_process,  # gaussian_process
                                            #tfd.MultivariateNormalDiag(loc=np.array([1.2,.2],dtype=np.float64), scale_diag=np.array([.2,.2],dtype=np.float64), name="theta"), # theta = (llambda, beta)
                                            tfd.Sample(tfd.Uniform(np.float64(0), np.float64(1)), sample_shape=2, name="theta"), # theta = (llambda, beta)
                                            lambda theta, gaussian_process: tfd.MultivariateNormalDiag(loc=(gaussian_process[..., observations_slice] + f_sim(observation_index_points[..., 0], theta[...,0], theta[...,1])), scale_identity_multiplier=np.sqrt(observation_noise_variance), name="observations")])

# Condition full joint posterior on observations
gp_model_conditioned = gp_model.experimental_pin(
    observations=observations)

sample = gp_model_conditioned.sample_unpinned(1)
prob = gp_model_conditioned.unnormalized_log_prob(sample)
constraining_bijector = gp_model_conditioned.experimental_default_event_space_bijector()

# NOTE: Can extend left-most dimension to run multiple chains but made runtime slower.
x_init = .5
current_state = [x_init*tf.ones_like(sample_component) for sample_component in sample]

scale = 0.001
step_size = [scale*tf.ones_like(sample_component)
             for sample_component in sample]


@tf.function(autograph=False, jit_compile=True)
def do_sampling():
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
        num_steps_between_results=1,
        trace_fn=lambda _, pkr: [pkr.inner_results.inner_results.step_size,
                                 pkr.inner_results.inner_results.log_accept_ratio])

print("Begin chain...")
samples, [step_size, log_accept_ratio] = do_sampling()
print("Chain finished.")

# Posterior predictive checking (PPC)
# NOTE: Could be wrong.
#gp_model_generative = gp_model.experimental_pin(length_scale=samples[0], amplitude=samples[1], gaussian_process=samples[2][:,0,:], theta=samples[3][:,0,:])
#generated_observations = gp_model_generative.sample_unpinned(1)

length_scale_samples = samples[0]
amplitude_samples = samples[1]
gp_samples = samples[2]
theta_samples = samples[3]

#generated_observations = generated_observations[0].numpy()

# Output
try:
    os.mkdir("output")
except:
    pass

# Output chain to file
np.save("output/length_scale_samples.npy", length_scale_samples.numpy())
np.save("output/amplitude_samples.npy", amplitude_samples.numpy())
np.save("output/theta_samples.npy", theta_samples.numpy())
np.save("output/gp_samples.npy", gp_samples.numpy())
np.save("output/log_accept_ratio.npy", log_accept_ratio.numpy())
#np.save("output/generated_observations.npy", generated_observations)

p_accept = tf.math.exp(tfp.math.reduce_logmeanexp(tf.minimum(
    log_accept_ratio, 0.)))
print(f"Acceptance ratio: {p_accept}")

# Gaussian process mean
_ = plt.figure()
gp_mean = np.mean(gp_samples, axis=0).reshape(-1)
gp_variance = np.var(gp_samples, axis=0).reshape(-1)
#
theta_mean=np.mean(theta_samples,axis=0).reshape(-1)
#
plt.plot(index_points[observations_slice], observations, 'o',  c='r')
xx = np.sort(index_points[..., 0])
plt.plot(xx, f_exp(xx), c='r')
#for i in range(3):
#    plt.plot(index_points[observations_slice],
#             generated_observations[i].reshape(-1), 'o', c='y', alpha=0.5)
plt.plot(index_points[predictions_slice], gp_mean[predictions_slice], '--', c='g')
#plt.plot(index_points[predictions_slice], (gp_mean + 2.0 * np.sqrt(gp_variance))[predictions_slice], c='r', alpha=0.5)
#plt.plot(index_points[predictions_slice], (gp_mean - 2.0 * np.sqrt(gp_variance))[predictions_slice], c='r', alpha=0.5)
plt.plot(index_points[predictions_slice], f_sim(prediction_index_points[..., 0], theta_mean[...,0], theta_mean[...,1]), '--', c='b')
plt.plot(index_points[predictions_slice], f_sim(prediction_index_points[..., 0], theta_mean[...,0], theta_mean[...,1]) + gp_mean[predictions_slice], c='b')
plt.fill_between(index_points[predictions_slice].reshape(-1),
                 f_sim(prediction_index_points[..., 0], theta_mean[...,0], theta_mean[...,1]) + gp_mean[predictions_slice] + 2.0 * np.sqrt(gp_variance[predictions_slice]),
                 f_sim(prediction_index_points[..., 0], theta_mean[...,0], theta_mean[...,1]) + gp_mean[predictions_slice] - 2.0 * np.sqrt(gp_variance[predictions_slice]),
                 alpha=0.2, color='b')
#plt.plot(index_points[predictions_slice], f_sim(prediction_index_points[..., 0], theta_mean[...,0], theta_mean[...,1]) + gp_mean[predictions_slice]
#         + 2.0 * np.sqrt(gp_variance[predictions_slice]), c='b', alpha=0.5)
#plt.plot(index_points[predictions_slice], f_sim(prediction_index_points[..., 0], theta_mean[...,0], theta_mean[...,1]) + gp_mean[predictions_slice]
#         - 2.0 * np.sqrt(gp_variance[predictions_slice]), c='b', alpha=0.5)
plt.savefig("output/gp_pred.png")
plt.clf()


# Scalar hyperparameters
_ = plt.figure()
plt.hist(length_scale_samples.numpy().reshape(-1), 30, density=True)
plt.xlabel(r"$L_{GP}$")
plt.savefig("output/length_scale.png")
plt.clf()

_ = plt.figure()
plt.hist(amplitude_samples.numpy().reshape(-1), 30, density=True)
plt.xlabel(r"$\sigma^2_{GP}$")
plt.savefig("output/amplitude.png")
plt.clf()

# Sim parameters
# llambda
ind_param=0
_ = plt.figure()
plt.hist(theta_samples.numpy()[...,ind_param].reshape(-1), 30, density=True)
plt.xlabel(r"$\lambda$")
plt.savefig("output/lambda.png")
plt.clf()
# beta
ind_param=1
_ = plt.figure()
plt.hist(theta_samples.numpy()[...,ind_param].reshape(-1), 30, density=True)
plt.xlabel(r"$\beta$")
plt.savefig("output/beta.png")
plt.clf()
