import os
import numpy as np
import string
#from matplotlib.colors import to_rgba as rgba
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns
import pandas as pd

############ LOAD TENSORFLOW MODEL FROM MODELDIR ############
#
MODELDIR = os.getcwd() # DIRECTORY CONTAINING generate_model.py
#
import importlib.util
spec = importlib.util.spec_from_file_location("generate_model", os.path.join(MODELDIR,"generate_model.py"))
generate_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generate_model)
IND_MODEL = generate_model.IND_MODEL
LOGSCALE_X = generate_model.LOGSCALE_X
f_sim = generate_model.f_sim # WARNING: only works for scalar 't'
observation_noise_variance = generate_model.observation_noise_variance

############ SET I/O ############
if IND_MODEL==0:
    OUTDIR = "output_higdon_png" # DIR WHERE TO
    INDIRS = ["output_higdon"] # DIRECORIES CONTAINING .npy FILES
    gp_model = generate_model.gp_model_HIG
    gp_model_conditioned = generate_model.gp_model_conditioned_HIG
elif IND_MODEL==1:
    OUTDIR = "output_koh_png" # DIR WHERE TO
    INDIRS = ["output_koh"]
    gp_model = generate_model.gp_model_KOH
    gp_model_conditioned = generate_model.gp_model_conditioned_KOH
elif IND_MODEL==2:
    OUTDIR = "output_ml_png" # DIR WHERE TO
    INDIRS = ["output_ml"]
    gp_model = generate_model.gp_model_ML
    gp_model_conditioned = generate_model.gp_model_conditioned_ML
elif IND_MODEL==3:
    OUTDIR = "output_f_png" # DIR WHERE TO
    INDIRS = ["output_f"]
    gp_model = generate_model.gp_model_F
    gp_model_conditioned = generate_model.gp_model_conditioned_F


try:
    os.mkdir(OUTDIR)
except FileExistsError:
    pass

############ LOAD DATA ############
INDIR = INDIRS[0]
with np.load(os.path.join(INDIR,"data_config_EXP.npz"), allow_pickle=True) as data:
    YObs_EXP = data['YObs_EXP']
    X_EXP = data['X_EXP']
    sliceObs_EXP = data['sliceObs_EXP'].item()
    slicePred_EXP = data['slicePred_EXP'].item()
    XTest_EXP = data['XTest_EXP']
    YTest_EXP = data['YTest_EXP']

with np.load(os.path.join(INDIR,"data_config_SIM.npz"), allow_pickle=True) as data:
    YObs_SIM = data['YObs_SIM']
    X_SIM = data['X_SIM']
    sliceObs_SIM = data['sliceObs_SIM'].item()
    slicePred_SIM = data['slicePred_SIM'].item()
    
XObs_EXP = X_EXP[sliceObs_EXP]
XPred_EXP = X_EXP[slicePred_EXP]
XObs_SIM = X_SIM[sliceObs_SIM]
XPred_SIM = X_SIM[slicePred_SIM]
XObs_SIM_x = XObs_SIM[...,0]
XPred_SIM_x = XPred_SIM[...,0]
#XObs = np.vstack((XObs0_SIM,XObs_SIM))[...,0]
vals = XTest_EXP
vec=XPred_EXP
v=np.abs(vec[:, np.newaxis] - vals)
indTest_slicePred = np.where(v==0)[0] # indices for test data relative to slicePred_EXP, i.e. X_EXP[slicePred_EXP][indTest_slicePred] = X_SIM[slicePred_SIM,0][indTest_slicePred] = XTest_EXP  <--->  assuming XPred_SIM[...,0]=XPred_EXP


## intermediate formal necessity.
#np.save(os.path.join(INDIRS[0],f"{gp_model._flat_resolve_names()[-1]}.npy"), YObs_EXP)

###### GET SAMPLES #####
for INDIR in INDIRS:
    try:
        log_accept_ratio = np.concatenate((log_accept_ratio,
                                           np.load(os.path.join(INDIR,"log_accept_ratio.npy"))), axis=0)
    except Exception as e:
        log_accept_ratio = np.load(os.path.join(INDIR,"log_accept_ratio.npy"))


def make_sample_chain(model, postfix=''):

    # reference model keys
    model_ref = generate_model.gp_model_HIG
    keys_ref = model_ref._flat_resolve_names()

    # actual model keys, pins
    keys = model._flat_resolve_names()
    try:
        pins = list(model.pins)
    except AttributeError:
        pins = list()
    # keys not present in actual model
    keys_ghost = list(set(keys_ref).difference(set(keys).union(set(pins))))

    samples_chain = {}
    for INDIR in INDIRS:
        try:
            for i in range(len(keys_ref)-1): # len -1 as to ignore the last key 'observations'
                key_ref = keys_ref[i]
                if key_ref in keys:
                    samples_chain[key_ref] = np.concatenate((
                        samples_chain[key_ref],
                        np.load(os.path.join(INDIR,f"{key_ref}{postfix}.npy"))), axis=0)
                elif key_ref in pins:
                    shape_ref = log_accept_ratio.shape + model_ref.sample()[i].numpy().shape
                    pin = model.pins[key_ref].numpy()
                    samples_chain[key_ref] = np.concatenate((
                        samples_chain[key_ref],
                        np.ones(shape_ref) * pin), axis=0)
                elif key_ref in keys_ghost:
                    shape_ref = log_accept_ratio.shape + model_ref.sample()[i].numpy().shape
                    samples_chain[key_ref] = np.concatenate((
                        samples_chain[key_ref],
                        np.empty(shape_ref) * np.nan), axis=0)
                else:
                    print("Error: Key not found:", key_ref)
            print("concatd arrays from:", INDIR)
            
        except Exception as e:
            for i in range(len(keys_ref)-1): # len -1 as to ignore the last key 'observations'
                key_ref = keys_ref[i]
                if key_ref in keys:
                    print('init key', key_ref)
                    samples_chain[key_ref] = np.load(os.path.join(INDIR,f"{key_ref}{postfix}.npy"))
                elif key_ref in pins:
                    shape_ref = log_accept_ratio.shape + model_ref.sample()[i].numpy().shape
                    pin = model.pins[key_ref].numpy()
                    samples_chain[key_ref] = np.ones(shape_ref) * pin
                elif key_ref in keys_ghost:
                    shape_ref = log_accept_ratio.shape + model_ref.sample()[i].numpy().shape
                    samples_chain[key_ref] = np.empty(shape_ref) * np.nan
                else:
                    print("Error: Key not found:", key_ref)
            print("initd arrays from:", INDIR)

    return samples_chain



# Posterior samples (MCMC)
model_conditioned = gp_model_conditioned
samples_mcmc = make_sample_chain(model_conditioned)

gp_samples_MD = samples_mcmc['GP_MD']
theta_samples = samples_mcmc['theta']
gp_samples_SIM = samples_mcmc['GP_SIM']
length_scale_samples_MD = samples_mcmc['length_scale_MD']
amplitude_samples_MD = samples_mcmc['amplitude_MD']
mean_samples_MD = samples_mcmc['mean_MD']
length_scale_samples_SIM = samples_mcmc['length_scale_SIM']
amplitude_samples_SIM = samples_mcmc['amplitude_SIM']
mean_samples_SIM = samples_mcmc['mean_SIM']
mu_samples_MD = np.column_stack((length_scale_samples_MD, amplitude_samples_MD, mean_samples_MD))
mu_samples_SIM = np.column_stack((np.squeeze(length_scale_samples_SIM), amplitude_samples_SIM, mean_samples_SIM)) # np.squeeze only if length_scale_samples_SIM.shape[-1]>1

NbDimT = theta_samples.shape[-1]
NbDimX = 1

# Prior samples
model_prior = gp_model
samples_prior = make_sample_chain(model_prior, postfix='_prior')

gp_samples_MD_prior = samples_prior['GP_MD']
gp_samples_SIM_prior = samples_prior['GP_SIM']
theta_samples_prior = samples_prior['theta']
length_scale_samples_MD_prior = samples_prior['length_scale_MD']
amplitude_samples_MD_prior = samples_prior['amplitude_MD']
mean_samples_MD_prior = samples_prior['mean_MD']
length_scale_samples_SIM_prior = samples_prior['length_scale_SIM']
amplitude_samples_SIM_prior = samples_prior['amplitude_SIM']
mean_samples_SIM_prior = samples_prior['mean_SIM']
mu_samples_MD_prior = np.column_stack((length_scale_samples_MD_prior, amplitude_samples_MD_prior, mean_samples_MD_prior))
mu_samples_SIM_prior = np.column_stack((np.squeeze(length_scale_samples_SIM_prior), amplitude_samples_SIM_prior, mean_samples_SIM_prior)) # np.squeeze only if length_scale_samples_SIM.shape[-1]>1


# PPC
if IND_MODEL == 0:
    GP_PPC = gp_model.experimental_pin(
        mean_MD=mean_samples_MD,
        amplitude_MD=amplitude_samples_MD,
        length_scale_MD=length_scale_samples_MD,
        GP_MD=gp_samples_MD,
        mean_SIM=mean_samples_SIM,
        amplitude_SIM=amplitude_samples_SIM,
        length_scale_SIM=length_scale_samples_SIM,
        theta=theta_samples,
        GP_SIM=gp_samples_SIM)
elif IND_MODEL == 1:
    GP_PPC = gp_model.experimental_pin(
        mean_MD=mean_samples_MD,
        amplitude_MD=amplitude_samples_MD,
        length_scale_MD=length_scale_samples_MD,
        GP_MD=gp_samples_MD,
        theta=theta_samples)
elif IND_MODEL == 2:
    GP_PPC = gp_model.experimental_pin(
        mean_MD=mean_samples_MD,
        amplitude_MD=amplitude_samples_MD,
        length_scale_MD=length_scale_samples_MD,
        GP_MD=gp_samples_MD)
elif IND_MODEL == 3:
    GP_PPC = gp_model.experimental_pin(
        theta=theta_samples)

nsamples_per_mcmc_sample = 100
gp_samples_PPC = GP_PPC.sample_unpinned(nsamples_per_mcmc_sample)[-1].numpy()

############ RECORD ############
NbObs_EXP = len(XObs_EXP)
NbPred = len(XPred_EXP)
NbObs_SIM = len(XObs_SIM)
num_results = len(log_accept_ratio)
p_accept = np.mean(np.exp(np.fmin(log_accept_ratio, 0.)))
# parameters
# SIM
theta_mean=np.mean(theta_samples, axis=tuple([i for i in range(theta_samples.ndim-1)]))
theta_std = np.squeeze(np.std(theta_samples, axis=0))
mean_mean_SIM=np.mean(mean_samples_SIM, axis=tuple([i for i in range(mean_samples_SIM.ndim-1)]))
mean_std_SIM = np.squeeze(np.std(mean_samples_SIM, axis=0))
amplitude_mean_SIM=np.mean(amplitude_samples_SIM, axis=tuple([i for i in range(amplitude_samples_SIM.ndim-1)]))
amplitude_std_SIM = np.squeeze(np.std(amplitude_samples_SIM, axis=0))
length_scale_mean_SIM=np.mean(length_scale_samples_SIM, axis=tuple([i for i in range(length_scale_samples_SIM.ndim-1)]))
length_scale_std_SIM = np.squeeze(np.std(length_scale_samples_SIM, axis=0))
# MD
mean_mean_MD=np.mean(mean_samples_MD, axis=tuple([i for i in range(mean_samples_MD.ndim-1)]))
mean_std_MD = np.squeeze(np.std(mean_samples_MD, axis=0))
amplitude_mean_MD=np.mean(amplitude_samples_MD, axis=tuple([i for i in range(amplitude_samples_MD.ndim-1)]))
amplitude_std_MD = np.squeeze(np.std(amplitude_samples_MD, axis=0))
length_scale_mean_MD=np.mean(length_scale_samples_MD, axis=tuple([i for i in range(length_scale_samples_MD.ndim-1)]))
length_scale_std_MD = np.squeeze(np.std(length_scale_samples_MD, axis=0))
print('theta_mean =', theta_mean)
print('mean_mean_SIM =', mean_mean_SIM)
print('amplitude_mean_SIM =', amplitude_mean_SIM)
print('length_scale_mean_SIM =', length_scale_mean_SIM)
print('mean_mean_MD =', mean_mean_MD)
print('amplitude_mean_MD =', amplitude_mean_MD)
print('length_scale_mean_MD =', length_scale_mean_MD)

with open(os.path.join(OUTDIR,"record.txt"), "w") as f:
    f.write(f"Nb of exp. obs.: {NbObs_EXP}\n")
    f.write(f"Nb of sim. obs.: {NbObs_SIM}\n")
    f.write(f"Nb of predictions: {NbPred}\n")
    f.write(f"Nb of MCMC samples: {num_results}\n")
    f.write(f"Nb of PPC samples per MCMC sample: {nsamples_per_mcmc_sample}\n")
    f.write(f"Acceptance ratio: {p_accept}\n")
    f.write(f"theta posterior mean: {theta_mean}\n")
    f.write(f"mean SIM posterior mean: {mean_mean_SIM}\n")
    f.write(f"amplitude SIM posterior mean: {amplitude_mean_SIM}\n")
    f.write(f"length scale SIM posterior mean: {length_scale_mean_SIM}\n")
    f.write(f"mean MD posterior mean: {mean_mean_MD}\n")
    f.write(f"amplitude MD posterior mean: {amplitude_mean_MD}\n")
    f.write(f"length scale MD posterior mean: {length_scale_mean_MD}\n")


def make_record(params, template):

    with open(template, "r") as f:
        t = string.Template(f.read())

    out = t.substitute(params)
    with open(os.path.join(OUTDIR, "record.tex"), "w") as f:
        f.write(out)

        
params = {"NbObs_EXP": NbObs_EXP,
          "NbObs_SIM": NbObs_SIM,
          "NbPred": NbPred,
          "num_results": num_results,
          "nsamples_per_mcmc_sample": nsamples_per_mcmc_sample,
          "p_accept": np.half(p_accept),
          'mean_mean_MD': np.half(mean_mean_MD),
          'amplitude_mean_MD': np.half(amplitude_mean_MD),
          'length_scale_mean_MD': np.half(length_scale_mean_MD),
          'mean_mean_SIM': np.half(mean_mean_SIM),
          'amplitude_mean_SIM': np.half(amplitude_mean_SIM),
          'length_scale_mean_SIM': np.half(length_scale_mean_SIM),
          'theta_mean': np.half(theta_mean),
          'mean_std_MD': np.half(mean_std_MD),
          'amplitude_std_MD': np.half(amplitude_std_MD),
          'length_scale_std_MD': np.half(length_scale_std_MD),
          'mean_std_SIM': np.half(mean_std_SIM),
          'amplitude_std_SIM': np.half(amplitude_std_SIM),
          'length_scale_std_SIM': np.half(length_scale_std_SIM),
          'theta_std': np.half(theta_std)}

pdf = generate_model.make_prior_mean_MD()
params['mean_MD_prior_dist'] = pdf.record['dist']
d=pdf.record['params']
try:
    d=list(d)+[list(np.half(v.numpy())) for v in d.values()]
except TypeError:
    d=list(d)+[np.half(v.numpy()) for v in d.values()]
except AttributeError:
    d=list(d)+[np.half(v) for v in d.values()]
params['mean_MD_prior_params'] = d
d = pdf.record['mean'].numpy()
params['mean_MD_prior_mean'] = np.half(d)
d = pdf.record['std'].numpy()
params['mean_MD_prior_std'] = np.half(d)

pdf = generate_model.make_prior_amplitude_MD()
params['amplitude_MD_prior_dist'] = pdf.record['dist']
d=pdf.record['params']
try:
    d=list(d)+[list(np.half(v.numpy())) for v in d.values()]
except TypeError:
    d=list(d)+[np.half(v.numpy()) for v in d.values()]
except AttributeError:
    d=list(d)+[np.half(v) for v in d.values()]
params['amplitude_MD_prior_params'] = d
d = pdf.record['mean'].numpy()
params['amplitude_MD_prior_mean'] = np.half(d)
d = pdf.record['std'].numpy()
params['amplitude_MD_prior_std'] = np.half(d)

pdf = generate_model.make_prior_length_scale_MD()
params['length_scale_MD_prior_dist'] = pdf.record['dist']
d=pdf.record['params']
try:
    d=list(d)+[list(np.half(v.numpy())) for v in d.values()]
except TypeError:
    d=list(d)+[np.half(v.numpy()) for v in d.values()]
except AttributeError:
    d=list(d)+[np.half(v) for v in d.values()]
params['length_scale_MD_prior_params'] = d
d = pdf.record['mean'].numpy()
params['length_scale_MD_prior_mean'] = np.half(d)
d = pdf.record['std'].numpy()
params['length_scale_MD_prior_std'] = np.half(d)

pdf = generate_model.make_prior_mean_SIM()
params['mean_SIM_prior_dist'] = pdf.record['dist']
d=pdf.record['params']
try:
    d=list(d)+[list(np.half(v.numpy())) for v in d.values()]
except TypeError:
    d=list(d)+[np.half(v.numpy()) for v in d.values()]
except AttributeError:
    d=list(d)+[np.half(v) for v in d.values()]
params['mean_SIM_prior_params'] = d
d = pdf.record['mean'].numpy()
params['mean_SIM_prior_mean'] = np.half(d)
d = pdf.record['std'].numpy()
params['mean_SIM_prior_std'] = np.half(d)

pdf = generate_model.make_prior_amplitude_SIM()
params['amplitude_SIM_prior_dist'] = pdf.record['dist']
d=pdf.record['params']
try:
    d=list(d)+[list(np.half(v.numpy())) for v in d.values()]
except TypeError:
    d=list(d)+[np.half(v.numpy()) for v in d.values()]
except AttributeError:
    d=list(d)+[np.half(v) for v in d.values()]
params['amplitude_SIM_prior_params'] = d
d = pdf.record['mean'].numpy()
params['amplitude_SIM_prior_mean'] = np.half(d)
d = pdf.record['std'].numpy()
params['amplitude_SIM_prior_std'] = np.half(d)

pdf = generate_model.make_prior_length_scale_SIM()
params['length_scale_SIM_prior_dist'] = pdf.record['dist']
d=pdf.record['params']
try:
    d=list(d)+[list(np.half(v.numpy())) for v in d.values()]
except TypeError:
    d=list(d)+[np.half(v.numpy()) for v in d.values()]
except AttributeError:
    d=list(d)+[np.half(v) for v in d.values()]
params['length_scale_SIM_prior_params'] = d
d = pdf.record['mean'].numpy()
params['length_scale_SIM_prior_mean'] = np.half(d)
d = pdf.record['std'].numpy()
params['length_scale_SIM_prior_std'] = np.half(d)

pdf = generate_model.make_prior_theta()
params['theta_prior_dist'] = pdf.record['dist']
d=pdf.record['params']
try:
    d=list(d)+[list(np.half(v.numpy())) for v in d.values()]
except TypeError:
    d=list(d)+[np.half(v.numpy()) for v in d.values()]
except AttributeError:
    d=list(d)+[np.half(v) for v in d.values()]
params['theta_prior_params'] = d
d = pdf.record['mean'].numpy()
params['theta_prior_mean'] = np.half(d)
d = pdf.record['std'].numpy()
params['theta_prior_std'] = np.half(d)

if IND_MODEL==0:
    make_record(params=params, template="record_higdon.template")
elif IND_MODEL==1:
    make_record(params=params, template="record_koh.template")
elif IND_MODEL==2:
    make_record(params=params, template="record_ml.template")
elif IND_MODEL==3:
    make_record(params=params, template="record_f.template")
    
    
############ PLOTS ############
def calc_xticks():
    xx = np.concatenate(( np.arange(2,12,2), np.array(20,ndmin=1) ))
    xlocs = np.log(xx)
    #plt.xticks(xlocs,xx)
    return xlocs, xx
#
# Gaussian process SIM+MD
#
gp_mean_MD = np.mean(gp_samples_MD, axis=tuple([i for i in range(gp_samples_MD.ndim-1)]))
#gp_variance_MD = np.squeeze(np.var(gp_samples_MD, axis=0))
gp_mean_SIM = np.mean(gp_samples_SIM, axis=tuple([i for i in range(gp_samples_SIM.ndim-1)]))
#gp_variance_SIM = np.squeeze(np.var(gp_samples_SIM, axis=0))
gp_mean_PPC = np.mean(gp_samples_PPC, axis=tuple([i for i in range(gp_samples_PPC.ndim-1)]))
#gp_variance_PPC = np.squeeze(np.var(np.mean(gp_samples_PPC, axis=0), axis=0)) # here, np.var(np.mean(...)), as multiple samples in axis=0 per posterior sample
# following works when XPred_SIM=XPred_EXP as usual
f_samples_SIM = f_sim(np.reshape(X_EXP,(-1,NbDimX)), theta_samples, LOGSCALE_X=LOGSCALE_X)
f_mean_SIM = np.mean(f_samples_SIM, axis=tuple([i for i in range(f_samples_SIM.ndim-1)]))
#f_variance_SIM = np.squeeze(np.var(f_samples_SIM, axis=0))
if IND_MODEL==0:
    gp_samples_PP = np.append(gp_samples_MD[...,sliceObs_EXP],gp_samples_MD[...,slicePred_EXP],axis = -1) + np.append(gp_samples_SIM[...,sliceObs_EXP],gp_samples_SIM[...,slicePred_SIM],axis = -1)
elif IND_MODEL==1:
    gp_samples_PP = np.squeeze(gp_samples_MD) + f_samples_SIM
elif IND_MODEL==2:
    gp_samples_PP = np.squeeze(gp_samples_MD)
elif IND_MODEL==3:
    gp_samples_PP = f_samples_SIM
gp_mean_PP = np.mean(gp_samples_PP, axis=tuple([i for i in range(gp_samples_PP.ndim-1)]))
#gp_variance_PP = np.squeeze(np.var(gp_samples_PP, axis=0))
#
_ = plt.figure()
plt.plot(XObs_EXP, YObs_EXP, 's',  c='black', label=r"training data")
#err_obs = 2.* np.sqrt(observation_noise_variance)
#plt.errorbar(XObs_EXP, YObs_EXP, yerr=err_obs, fmt='s', c='black', label=r"training data")
plt.plot(XTest_EXP, YTest_EXP, 's',  c='grey', label=r"test data")
if IND_MODEL == 0:
    plt.plot(XPred_EXP, gp_mean_SIM[slicePred_SIM], '-.', c='r', label=r"sim. model (post. mean)")
    #plt.plot(XTest_EXP, gp_mean_SIM[slicePred_SIM][indTest_slicePred], 'x', c='r', label=r"sim. model pred.")
elif IND_MODEL == 1:
    plt.plot(XPred_EXP, f_mean_SIM[slicePred_EXP], '-.', c='r', label=r"sim. model (post. mean)")
else:
    pass
plt.plot(XPred_EXP, gp_mean_PP[slicePred_EXP], '-', c='r', label=r"pred. model (post. mean)")
#plt.plot(XObs_EXP, gp_mean_PP[sliceObs_EXP], '-', c='r', label=r"pred. model (post. mean)")
#
q = 0.05 / 2.
upper = np.quantile(gp_samples_PP, 1.-q, axis=0).reshape(-1)
lower = np.quantile(gp_samples_PP, q, axis=0).reshape(-1)
plt.fill_between(XPred_EXP, lower[slicePred_EXP], upper[slicePred_EXP], alpha=0.1, color='tab:red', label=r"PP $95\%$")  # color=rgba('tab:red', alpha=.8)
#plt.fill_between(XObs_EXP, lower[sliceObs_EXP], upper[sliceObs_EXP], alpha=0.2, color='tab:red', label=r"PP $95\%$")
#
q = 0.25 / 2.
upper = np.quantile(gp_samples_PP, 1.-q, axis=0).reshape(-1)
lower = np.quantile(gp_samples_PP, q, axis=0).reshape(-1)
plt.fill_between(XPred_EXP, lower[slicePred_EXP], upper[slicePred_EXP], alpha=0.2, color='tab:red', label=r"PP $75\%$")
#
q = 0.05 / 2.
err_PPC = np.row_stack((
    np.quantile(np.mean(gp_samples_PPC, axis=0), q, axis=0),
    np.quantile(np.mean(gp_samples_PPC, axis=0), 1.-q, axis=0)
))
err_PPC = np.abs(gp_mean_PPC-err_PPC)
plt.errorbar(XObs_EXP, gp_mean_PPC[sliceObs_EXP], yerr=err_PPC[...,sliceObs_EXP], fmt='x', label=r"PPC $95\%$")
#
if LOGSCALE_X:
    xlocs,xx = calc_xticks(); plt.xticks(xlocs,xx)
plt.xlabel(r"Shear Rate")
plt.ylabel(r"Die Swell")
plt.legend()
plt.savefig(os.path.join(OUTDIR,"plot_GPMAP.pdf"))
plt.close()
#
# Gaussian process MD
_ = plt.figure()
plt.plot(XObs_EXP, YObs_EXP, 's',  c='black', label=r"training data")
plt.plot(XTest_EXP, YTest_EXP, 's',  c='grey', label=r"test data")
plt.plot(XPred_EXP, gp_mean_MD[slicePred_EXP], '-', c='r', label=r"pred. model (post. mean)")
plt.plot(XObs_EXP, gp_mean_MD[sliceObs_EXP], '-', c='r', label=r"pred. model (post. mean)")
#
q = 0.05 / 2.
upper_MD = np.quantile(gp_samples_MD, 1.-q, axis=0).reshape(-1)
lower_MD = np.quantile(gp_samples_MD, q, axis=0).reshape(-1)
plt.fill_between(XPred_EXP, lower_MD[slicePred_EXP], upper_MD[slicePred_EXP], alpha=0.1, color='tab:red', label=r"PP $95\%$")
#plt.fill_between(XObs_EXP, lower_MD[sliceObs_EXP], upper_MD[sliceObs_EXP], alpha=0.2, color='tab:red', label=r"PP $95\%$")
#
q = 0.25 / 2.
upper_MD = np.quantile(gp_samples_MD, 1.-q, axis=0).reshape(-1)
lower_MD = np.quantile(gp_samples_MD, q, axis=0).reshape(-1)
plt.fill_between(XPred_EXP, lower_MD[slicePred_EXP], upper_MD[slicePred_EXP], alpha=0.2, color='tab:red', label=r"PP $75\%$")
#
if LOGSCALE_X:
    xlocs,xx = calc_xticks(); plt.xticks(xlocs,xx)
plt.xlabel(r"Shear Rate")
plt.ylabel(r"Die Swell")
plt.legend()
plt.savefig(os.path.join(OUTDIR,"plot_GPMD.pdf"))
plt.close()
#
# Gaussian process SIM
_ = plt.figure()
plt.plot(XObs_EXP, YObs_EXP, 's',  c='black', label=r"training data")
plt.plot(XTest_EXP, YTest_EXP, 's',  c='grey', label=r"test data")
plt.plot(XObs_SIM_x, YObs_SIM, 'o',  c='r', alpha=0.2, label=r"sim. obs. $f(\theta)$")
plt.plot(XPred_SIM_x, gp_mean_SIM[slicePred_SIM], '-.', c='r', label=r"sim. model (post. mean)")
plt.plot(XPred_SIM_x, np.squeeze(f_sim(XPred_SIM,theta_mean,LOGSCALE_X=LOGSCALE_X)), '-', c='black', label=r"$f(\theta^\star)$")
#plt.plot(XObs_SIM_x, gp_mean_SIM[sliceObs_SIM], '-.', c='r', label=r"sim. model (post. mean)")
#
q = 0.05 / 2.
upper_SIM = np.quantile(gp_samples_SIM, 1.-q, axis=0).reshape(-1)
lower_SIM = np.quantile(gp_samples_SIM, q, axis=0).reshape(-1)
plt.fill_between(XPred_SIM_x, lower_SIM[slicePred_SIM], upper_SIM[slicePred_SIM], alpha=0.1, color='tab:red', label=r"PP $95\%$")
#plt.fill_between(XObs_SIM_x, lower_SIM[sliceObs_SIM], upper_SIM[sliceObs_SIM], alpha=0.2, color='tab:red', label=r"PP $95\%$")
#
q = 0.25 / 2.
upper_SIM = np.quantile(gp_samples_SIM, 1.-q, axis=0).reshape(-1)
lower_SIM = np.quantile(gp_samples_SIM, q, axis=0).reshape(-1)
plt.fill_between(XPred_SIM_x, lower_SIM[slicePred_SIM], upper_SIM[slicePred_SIM], alpha=0.2, color='tab:red', label=r"PP $75\%$")
#
if LOGSCALE_X:
    xlocs,xx = calc_xticks(); plt.xticks(xlocs,xx)
plt.xlabel(r"Shear Rate")
plt.ylabel(r"Die Swell")
plt.legend()
plt.savefig(os.path.join(OUTDIR,"plot_GPSIM.pdf"))
plt.close()
#
# f_SIM
_ = plt.figure()
plt.plot(XObs_EXP, YObs_EXP, 's',  c='black', label=r"training data")
plt.plot(XTest_EXP, YTest_EXP, 's',  c='grey', label=r"test data")
plt.plot(XPred_EXP, f_mean_SIM[slicePred_EXP], '-.', c='r', label=r"sim. model (post. mean)")
plt.plot(XPred_EXP, np.squeeze(f_sim(np.reshape(XPred_EXP,(-1,NbDimX)),theta_mean,LOGSCALE_X=LOGSCALE_X)), '-', c='black', label=r"$f(\theta^\star)$")
#
q = 0.05 / 2.
upper_SIM = np.quantile(f_samples_SIM, 1.-q, axis=0).reshape(-1)
lower_SIM = np.quantile(f_samples_SIM, q, axis=0).reshape(-1)
plt.fill_between(XPred_EXP, lower_SIM[slicePred_EXP], upper_SIM[slicePred_EXP], alpha=0.1, color='tab:red', label=r"PP $95\%$")
#
q = 0.25 / 2.
upper_SIM = np.quantile(f_samples_SIM, 1.-q, axis=0).reshape(-1)
lower_SIM = np.quantile(f_samples_SIM, q, axis=0).reshape(-1)
plt.fill_between(XPred_EXP, lower_SIM[slicePred_EXP], upper_SIM[slicePred_EXP], alpha=0.2, color='tab:red', label=r"PP $75\%$")
#
if LOGSCALE_X:
    xlocs,xx = calc_xticks(); plt.xticks(xlocs,xx)
plt.xlabel(r"Shear Rate")
plt.ylabel(r"Die Swell")
plt.legend()
plt.savefig(os.path.join(OUTDIR,"plot_fSIM.pdf"))
plt.close()

# Parameter plots
# from posterior (and prior) PDF
def plot_hist(x, xp, xstr, filename):
    #y = globals()[ystr].reshape(-1)
    #y_prior = globals()[ystr+'_prior'].reshape(-1)
    _ = plt.figure()
    plt.hist(x, 30, density=True)
    plt.hist(xp, 30, density=True, histtype='step')
    plt.xlabel(xstr)
    plt.savefig(os.path.join(OUTDIR,filename))
    plt.close()
def plot_mcmc(x, xstr, filename):
    # MCMC chain
    _ = plt.figure()
    plt.plot(np.arange(1.,len(x)+1.), x, '-')
    plt.xlabel(r"nb samples")
    plt.ylabel(xstr)
    plt.savefig(os.path.join(OUTDIR,filename))
    plt.close()

# parameter correlation plots
def plot_joint(x, xp, columns, filename):
    _ = plt.figure()
    df1 = pd.DataFrame(x, columns=columns).assign(PDF="posterior")
    df0 = pd.DataFrame(xp, columns=columns).assign(PDF="prior")
    #df = df1.append(df0, ignore_index=True)
    df = pd.concat((df1,df0), ignore_index=True)
    #f = sns.pairplot(df, kind="kde", hue="PDF")#, corner = True)
    #f.map_lower(sns.scatterplot,marker="+")
    #f.axes[0,0].set_xlim((0,4))
    #f = sns.jointplot(data=df, x=columns[0], y=columns[1], marker=".", s=0.1, hue="PDF", marginal_kws=dict(cut=0,bw_adjust=2))
    f = sns.jointplot(data=df, x=columns[0], y=columns[1], kind="kde", hue="PDF", joint_kws=dict(cut=0, bw_adjust=2), marginal_kws=dict(cut=0,bw_adjust=2))
    #f = sns.JointGrid(data=df, x=columns[0], y=columns[1])
    #f.plot_joint(sns.kdeplot, cut=0, bw_adjust=10)
    #f.plot_marginals(sns.kdeplot, cut=0, bw_adjust=10)
    #f.ax_joint.set_xlim((0,4))
    #f.ax_joint.set_ylim((0,40))
    plt.savefig(os.path.join(OUTDIR,filename))
    plt.close()


####
# MD
####
x=length_scale_samples_MD.reshape(-1)
xp = length_scale_samples_MD_prior.reshape(-1)
xstr = r'$L_{MD}$'
filename = 'length_scale_MD.pdf'
try:
    plot_hist(x=x, xp=xp, xstr=xstr, filename=filename)
except:
    pass
filename = 'length_scale_MD_MCMC.pdf'
try:
    plot_mcmc(x=x, xstr=xstr, filename=filename)
except:
    pass
#
# amplitude SIM
x=amplitude_samples_MD.reshape(-1)
xp = amplitude_samples_MD_prior.reshape(-1)
xstr = r'\sigma_{MD}'
filename = 'amplitude_MD.pdf'
try:
    plot_hist(x=x, xp=xp, xstr=xstr, filename=filename)
except:
    pass
filename = 'amplitude_MD_MCMC.pdf'
try:
    plot_mcmc(x=x, xstr=xstr, filename=filename)
except:
    pass
#
# mean MD
x=mean_samples_MD.reshape(-1)
xp = mean_samples_MD_prior.reshape(-1)
xstr = r'\sigma_{MD}'
filename = 'mean_MD.pdf'
try:
    plot_hist(x=x, xp=xp, xstr=xstr, filename=filename)
except:
    pass
filename = 'mean_MD_MCMC.pdf'
try:
    plot_mcmc(x=x, xstr=xstr, filename=filename)
except:
    pass
#    
# mu_MD
try:
    _ = plt.figure()
    columns=[r"$L_{MD}$", r"$\sigma_{MD}$", r"$\mu_{MD}$"]
    x = mu_samples_MD.reshape(-1,mu_samples_MD.shape[-1])
    xp = mu_samples_MD_prior.reshape(-1,mu_samples_MD_prior.shape[-1])
    df1 = pd.DataFrame(x, columns=columns).assign(PDF="posterior")
    df0 = pd.DataFrame(xp, columns=columns).assign(PDF="prior")
    #df = df1.append(df0, ignore_index=True)
    df = pd.concat((df1,df0), ignore_index=True)
    f = sns.PairGrid(data=df, hue="PDF", corner = True)
    f.map_lower(sns.kdeplot, cut=0, bw_adjust=10)#, levels=np.linspace(0.2,1.,5))
    f.map_diag(sns.kdeplot, cut=0, bw_adjust=2, fill=True)
    ax = np.fmin(np.quantile(x,0.01,axis=0),np.quantile(xp,0.01,axis=0))
    bx = np.fmax(np.quantile(x,0.99,axis=0),np.quantile(xp,0.99,axis=0))
    # from top to bottom (diag)
    try:
        f.axes[0,0].set_xlim((ax[0],bx[0])) # x axis of row 0 col. 0  --> LMD
        f.axes[0,0].set_ylim((ax[0],bx[0])) # y axis of row 0 col. 0
        #
        f.axes[1,1].set_xlim((ax[1],bx[1])) # x axis of row 1 col. 1  --> amplMD
        f.axes[1,1].set_ylim((ax[1],bx[1])) # y axis of row 1 col. 1
        #
        f.axes[2,2].set_xlim((ax[2],bx[2])) # x axis of row 2 col. 2  --> meanMD
        f.axes[2,2].set_ylim((ax[2],bx[2])) # y axis of row 2 col. 2
    except:
        pass
    plt.savefig(os.path.join(OUTDIR,"mu_MD.pdf"))
    plt.close()
except:
    pass
    #raise


#########
# SIM
#########
# length scale SIM
y0 = length_scale_samples_SIM[...,0].reshape(-1)
y1 = length_scale_samples_SIM[...,1].reshape(-1)
y2 = length_scale_samples_SIM[...,2].reshape(-1)
y0p = length_scale_samples_SIM_prior[...,0].reshape(-1)
y1p = length_scale_samples_SIM_prior[...,1].reshape(-1)
y2p = length_scale_samples_SIM_prior[...,2].reshape(-1)

x = y0
xp = y0p
xstr = r'$L_{x,SIM}$'
filename = 'length_scale_SIM.pdf'
try:
    plot_hist(x=x, xp=xp, xstr=xstr, filename=filename)
except:
    pass
filename = 'length_scale_SIM_MCMC.pdf'
try:
    plot_mcmc(x=x, xstr=xstr, filename=filename)
except:
    pass
#
# amplitude SIM
x=amplitude_samples_SIM.reshape(-1)
xp = amplitude_samples_SIM_prior.reshape(-1)
xstr = r'\sigma_{SIM}'
filename = 'amplitude_SIM.pdf'
try:
    plot_hist(x=x, xp=xp, xstr=xstr, filename=filename)
except:
    pass
filename = 'amplitude_SIM_MCMC.pdf'
try:
    plot_mcmc(x=x, xstr=xstr, filename=filename)
except:
    pass
#
# mean SIM
x=mean_samples_SIM.reshape(-1)
xp = mean_samples_SIM_prior.reshape(-1)
xstr = r'\sigma_{SIM}'
filename = 'mean_SIM.pdf'
try:
    plot_hist(x=x, xp=xp, xstr=xstr, filename=filename)
except:
    pass
filename = 'mean_SIM_MCMC.pdf'
try:
    plot_mcmc(x=x, xstr=xstr, filename=filename)
except:
    pass
#
# material parameters
#
# llambda
ind_param=0
x=theta_samples[...,ind_param].reshape(-1)
xp = theta_samples_prior[...,ind_param].reshape(-1)
xstr = r"$\lambda$"
filename = "lambda.pdf"
try:
    plot_hist(x=x, xp=xp, xstr=xstr, filename=filename)
except:
    pass
filename = "lambda_MCMC.pdf"
try:
    plot_mcmc(x=x, xstr=xstr, filename=filename)
except:
    pass
#
# beta
ind_param=1
x=theta_samples[...,ind_param].reshape(-1)
xp = theta_samples_prior[...,ind_param].reshape(-1)
xstr = r"$\beta$"
filename = "beta.pdf"
try:
    plot_hist(x=x, xp=xp, xstr=xstr, filename=filename)
except:
    pass
filename = "beta_MCMC.pdf"
try:
    plot_mcmc(x=x, xstr=xstr, filename=filename)
except:
    pass
#
# parameter correlation plots
#
# theta
columns=[r"$\lambda$", r"$\beta$"]
x = theta_samples.reshape(-1,NbDimT)
xp = theta_samples_prior.reshape(-1,NbDimT)
filename = "theta.pdf"
try:
    plot_joint(x=x, xp=xp, columns=columns, filename=filename)
except:
    print(f"failed creating joint plot: {filename}")
    pass
    #raise


#import sys
#sys.exit()
#
# mu_SIM
try:
    _ = plt.figure()
    columns=[r"$L_{GP,x}$", r"$L_{GP,\lambda}$", r"$L_{GP,\beta}$", r"$\sigma_{GP}$", r"$\mu_{GP}$"]
    x = mu_samples_SIM.reshape(-1,mu_samples_SIM.shape[-1])
    xp = mu_samples_SIM_prior.reshape(-1,mu_samples_SIM_prior.shape[-1])
    df1 = pd.DataFrame(x, columns=columns).assign(PDF="posterior")
    df0 = pd.DataFrame(xp, columns=columns).assign(PDF="prior")
    #df = df1.append(df0, ignore_index=True)
    df = pd.concat((df1,df0), ignore_index=True)
    #f = sns.pairplot(df, kind="kde", hue="PDF", corner = True)#, grid_kws=dict(cut=0,bw_adjust=2))
    #f.map_lower(sns.scatterplot,marker=".", s=0.1)
    f = sns.PairGrid(data=df, hue="PDF", corner = True)
    f.map_lower(sns.kdeplot, cut=0, bw_adjust=10)#, levels=np.linspace(0.2,1.,5))
    f.map_diag(sns.kdeplot, cut=0, bw_adjust=2, fill=True)
    ax = np.fmin(np.quantile(x,0.01,axis=0),np.quantile(xp,0.01,axis=0))
    bx = np.fmax(np.quantile(x,0.99,axis=0),np.quantile(xp,0.99,axis=0))
    try:
        f.axes[0,0].set_xlim((ax[0],bx[0])) # x axis of row 0 col. 0
        f.axes[0,0].set_ylim((ax[0],bx[0])) # y axis of row 0 col. 0
        #
        f.axes[1,1].set_xlim((ax[1],bx[1])) # x axis of row 1 col. 1
        f.axes[1,1].set_ylim((ax[1],bx[1])) # y axis of row 1 col. 1
        #
        f.axes[2,2].set_xlim((ax[2],bx[2])) # x axis of row 2 col. 2
        f.axes[2,2].set_ylim((ax[2],bx[2])) # y axis of row 2 col. 2
        #
        f.axes[3,3].set_xlim((ax[3],bx[3])) # x axis of row 3 col. 3
        f.axes[3,3].set_ylim((ax[3],bx[3])) # y axis of row 3 col. 3
        #
        f.axes[4,4].set_xlim((ax[4],bx[4])) # x axis of row 4 col. 4
        f.axes[4,4].set_ylim((ax[4],bx[4])) # y axis of row 4 col. 4
    except:
        pass
    #f = sns.jointplot(data=df, x=columns[0], y=columns[1], marker=".", s=0.1, hue="PDF", diag_kws=dict(cut=0,bw_adjust=2))
    #f.plot_joint(sns.kdeplot, cut=0, bw_adjust=10)
    #f.ax_joint.set_xlim((0,4))
    #f.ax_joint.set_ylim((0,40))
    plt.savefig(os.path.join(OUTDIR,"mu_SIM.pdf"))
    plt.close()
except:
    pass
#
# response surface
#
if 1==0:
    #O=np.ones_like(theta_samples)
    O=np.ones((theta_samples.shape[0],*X_SIM.shape))
    xt_samples = X_SIM*O # shape = (nb_samples, nb_gp_sim_abscissas_xt, NbDimX+NbDimT)
    xt_samples[...,slicePred_SIM,NbDimX:NbDimX+NbDimT] = xt_samples[...,slicePred_SIM,NbDimX:NbDimX+NbDimT]+theta_samples
    xt_samples[...,sliceObs_EXP,NbDimX:NbDimX+NbDimT] = xt_samples[...,sliceObs_EXP,NbDimX:NbDimX+NbDimT]+theta_samples
    #
    for ind_param in range(1,NbDimT+1):
        _ = plt.figure()
        ax = _.add_subplot(projection='3d')
        xs = xt_samples[...,0]
        ys = xt_samples[...,ind_param]
        zs = gp_samples_SIM
        #ax.scatter(xs[...,sliceObs_EXP], ys[...,sliceObs_EXP], np.ones_like(xs[...,sliceObs_EXP])*YObs_EXP, marker='.', alpha=0.2, s=2, c='g', label=r"exp. obs.")
        #ax.scatter(xs[...,sliceObs_SIM], ys[...,sliceObs_SIM], np.ones_like(xs[...,sliceObs_SIM])*YObs_SIM, marker='o', alpha=0.2, s=2, c='r', label=r"sim. obs.")
        #ax.scatter(xs, ys, zs, marker='.', alpha=0.2, s=1, label=r"sim. pred.")
        #xs_obs = XObs_EXP.flatten()
        #ys_obs = np.ones_like(xs_obs)*theta_mean[...,ind_param-1]
        #zs_obs = YObs_EXP.flatten()
        #ax.scatter(xs_obs, ys_obs, zs_obs, marker='o', alpha=1, s=2, c='g', label=r"exp. obs.")
        xs_obs = XObs_SIM[...,0].flatten()
        ys_obs = XObs_SIM[...,ind_param].flatten()
        zs_obs = YObs_SIM.flatten()
        ax.scatter(xs_obs, ys_obs, zs_obs, marker='o', alpha=1, s=2, c='r', label=r"sim. obs.")
        xs_pred = xs[...,slicePred_SIM].flatten()
        ys_pred = ys[...,slicePred_SIM].flatten()
        zs_pred = zs[...,slicePred_SIM].flatten()
        ax.scatter(xs_pred, ys_pred, zs_pred, marker='.', alpha=0.1, s=0.1, c='b', label=r"sim. pred.")
        #ax.scatter(xs[...,sliceObs_SIM], ys[...,sliceObs_SIM], zs[...,sliceObs_SIM], marker='o', alpha=0.2, s=2, c='r', label=r"sim. obs.")
        #ax.scatter(xs[...,slicePred_SIM], ys[...,slicePred_SIM], zs[...,slicePred_SIM], marker='.', alpha=0.2, s=1, label=r"sim. pred.")
        ax.set_xlabel(r'shear rate')
        if ind_param == 1:
            ax.set_ylabel(r'$\lambda$')
        elif ind_param == 2:
            ax.set_ylabel(r'$\beta$')
        else:
            pass
        #
        ax.set_zlabel(r"Die Swell")
        plt.legend()
        plt.savefig(os.path.join(OUTDIR,f"plot_GPSIM_surf_{ind_param}.pdf"), dpi=200)
        print(f"saved figure to file {OUTDIR}/plot_GPSIM_surf_{ind_param}.pdf")
        plt.close()
