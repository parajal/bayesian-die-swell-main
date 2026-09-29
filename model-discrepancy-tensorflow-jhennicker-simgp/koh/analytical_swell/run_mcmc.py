from generate_model import *

# MCMC parameters
num_results = 1000
num_burnin_steps = int(num_results*0.1)

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

print("Begin chain on posterior PDF...")
nchains=1
samples, [step_size, log_accept_ratio] = do_sampling(model=gp_model_conditioned, nchains=nchains, num_results=num_results, num_burnin_steps=num_burnin_steps)
print("Chain finished on posterior PDF.")

############ SAVING OUTPUT TO FILES ############

#amplitude_samples_MD = samples[0]
#length_scale_samples_MD = samples[1]
mu_samples = samples[0]
gp_samples_MD = samples[1]
theta_samples = samples[2]

try:
    os.mkdir("output")
except:
    pass

np.savez("output/data_config.npz", YObs_EXP.numpy(), tf.reshape(X_EXP,-1).numpy(), sliceObs_EXP, slicePred_EXP)
#np.save("output/amplitude_samples_MD.npy", amplitude_samples_MD.numpy())
#np.save("output/length_scale_samples_MD.npy", length_scale_samples_MD.numpy())
np.save("output/mu_samples.npy", mu_samples.numpy())
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
#theta_mean=tnp.mean(theta_samples,axis=range(theta_samples.ndim-1))
#print('theta_mean =', theta_mean)
#mu_mean=tnp.mean(mu_samples,axis=range(mu_samples.ndim-1))
#print('mu_mean =', mu_mean)
#amplitude_mean_MD=tf.reshape( tnp.mean(tnp.mean(amplitude_samples_MD,axis=0),axis=0), -1)
#print('amplitude_mean_MD =', amplitude_mean_MD)
#length_scale_mean_MD=tf.reshape( tnp.mean(tnp.mean(length_scale_samples_MD,axis=0),axis=0), -1)
#print('length_scale_mean_MD =', length_scale_mean_MD)

# PREDICTION MODEL
#gpPred_SIM = make_GP_SIM(theta_mean, length_scale_mean_SIM, amplitude_mean_SIM, loc=gp_mean_SIM)
#gpPred_MD = make_GP_MD(length_scale_mean_MD, amplitude_mean_MD, loc=gp_mean_MD)
#gp_pred = tfd.JointDistributionSequential([gp_pred_SIM, gp_pred_MD])
#print("Begin prediction chain...")
#nchains=2
#gp_pred_samples, [gp_pred_step_size, gp_pred_log_accept_ratio] = do_sampling(model=gp_pred, nchains=nchains, num_results=num_results, num_burnin_steps=num_burnin_steps)
#print("Prediction chain finished.")
#
#GP_PPC = gp_model.experimental_pin(
#    #amplitude_MD=amplitude_samples_MD,
#    #length_scale_MD=length_scale_samples_MD,
#    mu=mu_samples,
#    GP_MD=gp_samples_MD,
#    theta=theta_samples)
#
#nsamples_per_mcmc_sample = 100
#gp_samples_PPC = GP_PPC.sample_unpinned(nsamples_per_mcmc_sample)[0]
#np.save("output/gp_samples_PPC.npy", gp_samples_PPC.numpy())
#gp_mean_PPC = tnp.mean(gp_samples_PPC, axis=range(gp_samples_PPC.ndim-1))
#gp_variance_PPC = tnp.squeeze(tnp.var(tnp.mean(gp_samples_PPC, axis=0), axis=0))
#err = tnp.sqrt(tnp.sum((gp_mean_PPC-YObs_EXP)**2))
#print("err PCC:", err)
    
