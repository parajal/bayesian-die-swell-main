from generate_model import *

#### SET MODEL TO RUN ####
if IND_MODEL==0:
    OUTDIR = "output_higdon" # DIR WHERE TO
    model=gp_model_conditioned_HIG
    print('RUN HIGDON')
elif IND_MODEL==1:
    OUTDIR = "output_koh" # DIR WHERE TO
    model=gp_model_conditioned_KOH
    print('RUN KOH')
elif IND_MODEL==2:
    OUTDIR = "output_ml" # DIR WHERE TO
    model=gp_model_conditioned_ML
    print('RUN ML')
elif IND_MODEL==3:
    OUTDIR = "output_f" # DIR WHERE TO
    model=gp_model_conditioned_F
    print('RUN F')

###
try:
    os.mkdir(OUTDIR)
except FileExistsError:
    pass
except:
    raise


# MCMC parameters
num_results = 10000
num_burnin_steps = 100 #int(num_results*0.1)


############ MCMC SAMPLING ############
@tf.function(autograph=False, jit_compile=True)
def do_sampling(model, sample, num_results, num_burnin_steps, step_size):
    
    prob = model.unnormalized_log_prob(sample)
    constraining_bijector = model.experimental_default_event_space_bijector()
    # NOTE: Can extend left-most dimension to run multiple chains but made runtime slower:
    #x_init = 1.
    #current_state = [x_init*tf.ones_like(sample_component) for sample_component in sample]
    current_state = sample
    #s=model.sample_unpinned((100,nchains))
    #sample = [tnp.mean(sample_component,axis=0) for sample_component in s]
    #prob = model.unnormalized_log_prob(sample)
    #constraining_bijector = model.experimental_default_event_space_bijector()
    #current_state = sample
    #scale = 0.001
    #step_size = [scale*tnp.var(sample_component,axis=0) for sample_component in s]
    #
    if True:
        nuts = tfp.mcmc.NoUTurnSampler(target_log_prob_fn=model.unnormalized_log_prob,
                                       step_size=step_size)
        #nuts = tfp.experimental.mcmc.PreconditionedNoUTurnSampler(target_log_prob_fn=model.unnormalized_log_prob,
        #                               step_size=step_size) # TODO: define appropriate momentum_distribution for precond.
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

    else:
        hmc = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=model.unnormalized_log_prob,
            num_leapfrog_steps=3,
            step_size=step_size)
        transformed_hmc = tfp.mcmc.TransformedTransitionKernel(
            hmc, bijector=constraining_bijector)
        adapted_transformed_hmc = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=transformed_hmc, num_adaptation_steps=int(num_burnin_steps * .8))
        samples, pkr = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state=current_state,
            kernel=adapted_transformed_hmc,
            num_steps_between_results=0,
            trace_fn=lambda _, pkr: [pkr.inner_results.inner_results.accepted_results.step_size, pkr.inner_results.inner_results.log_accept_ratio])

    return samples, pkr




print("Begin chain...")
nchains=1
nsamples_for_prior_mean = 100
try:
    #sample = model.sample(nchains)
    samples_for_prior_mean = model.sample((nsamples_for_prior_mean, nchains))
except AttributeError:
    #sample = model.sample_unpinned(nchains)
    samples_for_prior_mean = model.sample_unpinned((nsamples_for_prior_mean, nchains))
except:
    raise
#scale = 0.1
#step_size = [scale*tf.ones_like(sample_component) for sample_component in sample]
scale = tnp.float64(.4)
step_size = [scale*tnp.std(sample_component, axis=0) for sample_component in samples_for_prior_mean] # anisotropic adapted step_size init
sample_init = [tnp.mean(sample_component, axis=0) for sample_component in samples_for_prior_mean] # high probability init
#sample_init = [s+ds/4. for s,ds in zip(sample_init,step_size)]
print('init mcmc step size:', step_size)
samples, [step_size, log_accept_ratio] = do_sampling(model=model, sample=sample_init, num_results=num_results, num_burnin_steps=num_burnin_steps, step_size=step_size)
print("Chain finished.")

############ SAVING OUTPUT TO FILES ############

#d = {}
for i in range(len(model._flat_resolve_names())):
    #d[model._flat_resolve_names()[i]] = samples[i]
    np.save(os.path.join(OUTDIR,f"{model._flat_resolve_names()[i]}.npy"), samples[i].numpy())


np.savez(os.path.join(OUTDIR,"data_config_EXP.npz"),
         YObs_EXP=YObs_EXP.numpy(),
         X_EXP=tf.reshape(X_EXP,-1).numpy(),
         sliceObs_EXP=sliceObs_EXP,
         slicePred_EXP=slicePred_EXP,
         XTest_EXP=tf.reshape(XTest_EXP,-1).numpy(),
         YTest_EXP=tf.reshape(YTest_EXP,-1).numpy())
np.savez(os.path.join(OUTDIR,"data_config_SIM.npz"),
         YObs_SIM=YObs_SIM.numpy(),
         X_SIM=X_SIM.numpy(),
         sliceObs_SIM=sliceObs_SIM,
         slicePred_SIM=slicePred_SIM)

np.save(os.path.join(OUTDIR,"log_accept_ratio.npy"), log_accept_ratio.numpy())
p_accept = tf.math.exp(tfp.math.reduce_logmeanexp(tf.minimum(log_accept_ratio, 0.)))
print(f"Acceptance ratio: {p_accept}")

#### samples from prior PDF
num_samples_prior = num_results # --> number of independent prior samples 
#### save samples to files
for i in range(len(model.distribution._flat_resolve_names())):
    
    if len(model.distribution._dist_fn_args[i]): # dependent pdfs (only GPs, here) not needed
        if True:
            try:
                import shutil
                src = os.path.join(OUTDIR,f"{model.distribution._flat_resolve_names()[i]}.npy")
                dst = os.path.join(OUTDIR,f"{model.distribution._flat_resolve_names()[i]}_prior.npy")
                shutil.copyfile(src, dst)
            finally:
                continue
        else:
            continue
        
    pdf_i = model.distribution._dist_fn[i]().distribution
    si = pdf_i.sample(num_results)
    if True: # make duplicates
        oi = np.ones_like(si.shape, dtype=int)
        oi[0] = int(num_results/num_samples_prior)
        si = np.tile(si,oi)
        
    np.save(os.path.join(OUTDIR,f"{model.distribution._flat_resolve_names()[i]}_prior.npy"), si)    
