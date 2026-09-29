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

# Gaussian process SIM+MD
#_ = plt.figure()
#gp_mean_SIM = tf.reshape( tnp.mean(tnp.mean(gp_samples_SIM, axis=0), axis=0), -1)
#gp_variance_SIM = tf.reshape( tnp.mean(tnp.var(gp_samples_SIM, axis=0), axis=0), -1)
#gp_mean_MD = tf.reshape( tnp.mean(tnp.mean(gp_samples_MD, axis=0), axis=0), -1)
#gp_variance_MD = tf.reshape( tnp.mean(tnp.var(gp_samples_MD, axis=0), axis=0), -1)
#plt.plot(X_SIM[sliceObs_EXP,0], YObs_EXP, 'o',  c='r')
#plt.plot(X_SIM[sliceObs_SIM,0], YObs_SIM, 'o',  c='r', alpha=0.5)
#plt.plot(X_SIM[slicePred_SIM,0], gp_mean_MD[slicePred_EXP]+gp_mean_SIM[slicePred_SIM], '-', c='r')
#plt.savefig("output/plot_GPMAP.png")
#plt.clf()

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
#    amplitude_MD=amplitude_samples_MD,
#    length_scale_MD=length_scale_samples_MD,
#    GP_MD=gp_samples_MD,
#    amplitude_SIM=amplitude_samples_SIM,
#    scale_diag_SIM=length_scale_samples_SIM,
#    theta=theta_samples,
#    GP_SIM=gp_samples_SIM)
#nsamples_per_mcmc_sample = 100
#gp_samples_PPC = GP_PPC.sample_unpinned(nsamples_per_mcmc_sample)[0]
#np.save("output/gp_samples_PPC.npy", gp_samples_PPC.numpy())
## plot PPC
#gp_mean_PPC = tnp.mean(gp_samples_PPC, axis=range(gp_samples_PPC.ndim-1))
#gp_variance_PPC = tnp.squeeze(tnp.var(tnp.mean(gp_samples_PPC, axis=0), axis=0))
#err = tnp.sqrt(tnp.sum((gp_mean_PPC-ZObs)**2))
#print("err PCC:", err)
## on YObs_EXP
#_ = plt.figure()
#plt.plot(XObs_EXP, YObs_EXP, 'o',  c='black', label=r"training data")
#plt.plot(XObs_EXP[...,0], gp_mean_PPC[sliceObs_EXP], 'x',  c='black', label=r"PPC")
#err_PPC = 2.0 * np.sqrt(gp_variance_PPC[sliceObs_EXP])
#plt.errorbar(XObs_EXP[...,0], gp_mean_PPC[sliceObs_EXP], yerr=err_PPC, fmt='x', c='black', label=r"PPC error")
#plt.legend()
#plt.savefig("output/plot_PPC_EXP.png")
#plt.clf()
## on YObs_SIM
#_ = plt.figure()
#plt.plot(XObs_SIM[...,0], YObs_SIM, 'o',  c='black', label=r"training data")
#plt.plot(XObs_SIM[...,0], gp_mean_PPC[sliceObs_SIM], 'x',  c='black', label=r"PPC")
#err_PPC = 2.0 * np.sqrt(gp_variance_PPC[sliceObs_SIM])
#plt.errorbar(XObs_SIM[...,0], gp_mean_PPC[sliceObs_SIM], yerr=err_PPC, fmt='x', c='black', label=r"PPC error")
#plt.legend()
#plt.savefig("output/plot_PPC_SIM.png")
#plt.clf()


# RUN MCMC ON PRIOR AND SAVE CHAIN
#ind_mcmc_prior = 0
#if ind_mcmc_prior == 1:
#    print("Begin chain on prior PDF...")
#    nchains=1
#    samples_prior, [step_size_prior, log_accept_ratio_prior] = do_sampling(model=gp_model, nchains=nchains, num_results=num_results, num_burnin_steps=num_burnin_steps)
#    print("Chain finished on prior PDF.")
#    print("Acceptance ratio:", tf.math.exp(tfp.math.reduce_logmeanexp(tf.minimum(log_accept_ratio_prior, 0.))))
#    np.save("output/log_accept_ratio_prior.npy", log_accept_ratio_prior.numpy())
#else:
#    samples_prior = gp_model.sample(num_results)

#samples_prior = gp_model.sample(num_results)
#np.save("output/amplitude_samples_MD_prior.npy", samples_prior[0].numpy())
#np.save("output/length_scale_samples_MD_prior.npy", samples_prior[1].numpy())
#np.save("output/gp_samples_MD_prior.npy", samples_prior[2].numpy())
#np.save("output/amplitude_samples_SIM_prior.npy", samples_prior[3].numpy())
#np.save("output/length_scale_samples_SIM_prior.npy", samples_prior[4].numpy())
#np.save("output/theta_samples_prior.npy", samples_prior[5].numpy())
#np.save("output/gp_samples_SIM_prior.npy", samples_prior[6].numpy())
