import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

############ SET I/O ############
OUTDIR = "output" # DIR WHERE TO
try:
    os.mkdir(OUTDIR)
except FileExistsError:
    pass
except:
    raise

INDIRS = ["output"] # DIRECORIES CONTAINING .npy FILES

MODELDIR = os.path.join(INDIRS[0],'..') # DIRECTORY CONTAINING generate_model.py

############ LOAD POSTERIOR PDF ############
import importlib.util
spec = importlib.util.spec_from_file_location("generate_model", os.path.join(MODELDIR,"generate_model.py"))
generate_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generate_model)
gp_model = generate_model.gp_model
#f_sim = generate_model.f_sim

############ LOAD DATA ############
INDIR = INDIRS[0]
with np.load(os.path.join(INDIR,"data_config.npz"), allow_pickle=True) as data:
    YObs_EXP = data['arr_0']
    X_EXP = data['arr_1']
    sliceObs_EXP = data['arr_2'].item()
    slicePred_EXP = data['arr_3'].item()
    
XObs_EXP = X_EXP[sliceObs_EXP]
XPred_EXP = X_EXP[slicePred_EXP]
ZObs = YObs_EXP

for INDIR in INDIRS:
    try:
        mu_samples = np.concatenate((mu_samples,
                                     np.load(os.path.join(INDIR,"mu_samples.npy"))), axis=0)
        gp_samples_MD = np.concatenate((gp_samples_MD,
                                        np.load(os.path.join(INDIR,"gp_samples_MD.npy"))), axis=0)
        theta_samples = np.concatenate((theta_samples,
                                        np.load(os.path.join(INDIR,"theta_samples.npy"))), axis=0)
        log_accept_ratio = np.concatenate((log_accept_ratio,
                                           np.load(os.path.join(INDIR,"log_accept_ratio.npy"))), axis=0)
        print("concat arrays from:", INDIR)
    except Exception as e:
        print("init arrays from:", INDIR)
        mu_samples = np.load(os.path.join(INDIR,"mu_samples.npy"))
        gp_samples_MD = np.load(os.path.join(INDIR,"gp_samples_MD.npy"))
        theta_samples = np.load(os.path.join(INDIR,"theta_samples.npy"))
        log_accept_ratio = np.load(os.path.join(INDIR,"log_accept_ratio.npy"))
        #p_accept = tf.math.exp(tfp.math.reduce_logmeanexp(tf.minimum(log_accept_ratio, 0.)))
        #print(f"Acceptance ratio: {p_accept}")

amplitude_samples_MD = mu_samples[...,0]
length_scale_samples_MD = mu_samples[...,1]

# Prior
nsamples_prior = len(mu_samples)  # needed until fixed PDE normalization in joint-/pairplot
samples_prior = gp_model.sample(nsamples_prior)
mu_samples_prior = samples_prior[0].numpy()
gp_samples_MD_prior = samples_prior[1].numpy()
theta_samples_prior = samples_prior[2].numpy()
amplitude_samples_MD_prior = mu_samples_prior[...,0]
length_scale_samples_MD_prior = mu_samples_prior[...,1]

# PPC
GP_PPC = gp_model.experimental_pin(
    #amplitude_MD=amplitude_samples_MD,
    #length_scale_MD=length_scale_samples_MD,
    mu=mu_samples,
    GP_MD=gp_samples_MD,
    theta=theta_samples)

nsamples_per_mcmc_sample = 100
gp_samples_PPC = GP_PPC.sample_unpinned(nsamples_per_mcmc_sample)[0].numpy()


############ SIMULATION MODEL ############

# simulator
if 1 == 1:
    #from koh import f_sim
    def f_sim(x, t):
        R_Die = 1.
        shearRate = x
        llambda = t[...,0]
        beta = t[..., 1]
        Sw = (1-beta)*llambda*shearRate
        return ( (1+Sw**2/2)**(1/6) + .13 )
else:
    # SIM=0
    theta_samples = np.array((0.,0.))
    def f_sim(x, t):
        return x*0.

############ PLOTS ############
# parameters
theta_mean=np.mean(theta_samples, axis=tuple([i for i in range(theta_samples.ndim-1)]))
print('theta_mean =', theta_mean)
amplitude_mean_MD=np.mean(amplitude_samples_MD, axis=tuple([i for i in range(amplitude_samples_MD.ndim-1)]))
print('amplitude_mean_MD =', amplitude_mean_MD)
length_scale_mean_MD=np.mean(length_scale_samples_MD, axis=tuple([i for i in range(length_scale_samples_MD.ndim-1)]))
print('length_scale_mean_MD =', length_scale_mean_MD)

# Gaussian process SIM+MD
_ = plt.figure()
gp_mean_MD = np.mean(gp_samples_MD, axis=tuple([i for i in range(gp_samples_MD.ndim-1)]))
gp_variance_MD = np.var(gp_samples_MD, axis=tuple([i for i in range(gp_samples_MD.ndim-1)]))
gp_mean_PPC = np.mean(gp_samples_PPC, axis=tuple([i for i in range(gp_samples_PPC.ndim-1)]))
gp_variance_PPC = np.squeeze(np.var(np.mean(gp_samples_PPC, axis=0), axis=0))
f_mean_SIM = np.mean(f_sim(X_EXP, theta_samples), axis=tuple([i for i in range(f_sim(X_EXP, theta_samples).ndim-1)]))
f_variance_SIM = np.var(f_sim(X_EXP, theta_samples), axis=tuple([i for i in range(f_sim(X_EXP, theta_samples).ndim-1)]))
gp_mean_PP = np.mean(gp_samples_MD.reshape(f_sim(X_EXP, theta_samples).shape) + f_sim(X_EXP, theta_samples),axis=0)
gp_variance_PP = np.var(gp_samples_MD.reshape(f_sim(X_EXP, theta_samples).shape) + f_sim(X_EXP, theta_samples),axis=0)
# SBR data 0
#xx0 = np.array((2.0000,2.6000,3.3800,4.3940,5.7120,7.4260,21.2090))
xx1 = np.array((9.6540,12.5500,16.3150))#,21.2090))
R_Die = 1.
#yy0 = np.array((2.7587,2.8780,2.9404,2.9918,3.0718,3.1419,3.1023)) / (2.*R_Die)
#yy1 = np.array((3.0958,3.0669,3.0660)) / (2.*R_Die)#,3.1023)) / (2.*R_Die)
#plt.plot(xx0, yy0, 'o',  c='r', label=r"training data");
#plt.plot(xx1, yy1, 'o',  c='tab:grey', label=r"test data")
#
plt.plot(X_EXP[sliceObs_EXP], YObs_EXP, 'o',  c='r', label=r"training data")
plt.plot(X_EXP[slicePred_EXP], f_mean_SIM[slicePred_EXP], '-.', c='r', label=r"sim. model (post. mean)")
#plt.plot(X_EXP[slicePred_EXP], gp_mean_MD[slicePred_EXP] + f_sim(X_EXP[slicePred_EXP], theta_mean), '-', c='r')
plt.plot(X_EXP[slicePred_EXP], gp_mean_MD[slicePred_EXP] + f_mean_SIM[slicePred_EXP], '-', c='r', label=r"pred. model (post. mean)")
plt.plot(X_EXP[sliceObs_EXP], gp_mean_MD[sliceObs_EXP] + f_mean_SIM[sliceObs_EXP], '-', c='r')
#plt.plot(X_EXP[sliceObs_EXP], gp_mean_MD[sliceObs_EXP], '-', c='r')

#plt.plot(X_EXP[sliceObs_EXP], gp_mean_PPC[sliceObs_EXP], '-', c='b')
upper = gp_mean_PP + 2.0 * np.sqrt(gp_variance_PP)
lower = gp_mean_PP - 2.0 * np.sqrt(gp_variance_PP)
#plt.fill_between(X_EXP[sliceObs_EXP], lower[sliceObs_EXP], upper[sliceObs_EXP],
#                 alpha=0.2, color='tab:red')
plt.fill_between(X_EXP[slicePred_EXP], lower[slicePred_EXP], upper[slicePred_EXP],
                 alpha=0.2, color='tab:red', label=r"PP Credibility interval, $\pm 2 \sigma$")
err_PPC = 2.0 * np.sqrt(gp_variance_PPC)
plt.errorbar(X_EXP[sliceObs_EXP], gp_mean_PPC, yerr=err_PPC, fmt='x', label=r"PPC")
plt.legend()
plt.savefig(os.path.join(OUTDIR,"plot_GPMAP.png"))
plt.close()
#
# only SIM
_ = plt.figure()
plt.plot(X_EXP[sliceObs_EXP], YObs_EXP, 'o',  c='r', label=r"training data")
plt.plot(X_EXP[slicePred_EXP], f_mean_SIM[slicePred_EXP], '-.', c='r', label=r"pred. model (post. mean)")
plt.plot(X_EXP[sliceObs_EXP], f_mean_SIM[sliceObs_EXP], '-.', c='r')
plt.plot(X_EXP[slicePred_EXP], f_sim(X_EXP[slicePred_EXP], theta_mean), '-', c='black', label=r"$f(\theta^\star)$")
upper_SIM = f_mean_SIM + 2.0 * np.sqrt(f_variance_SIM)
lower_SIM = f_mean_SIM - 2.0 * np.sqrt(f_variance_SIM)
plt.fill_between(X_EXP[slicePred_EXP], lower_SIM[slicePred_EXP], upper_SIM[slicePred_EXP],
                 alpha=0.2, color='tab:blue', label=r"PP Credibility interval, $\pm 2 \sigma$")
plt.fill_between(X_EXP[sliceObs_EXP], lower_SIM[sliceObs_EXP], upper_SIM[sliceObs_EXP],
                 alpha=0.2, color='tab:blue')
plt.legend()
plt.savefig(os.path.join(OUTDIR,"plot_SIM.png"))
plt.close()
#
# Gaussian process MD
_ = plt.figure()
plt.plot(XObs_EXP, YObs_EXP, 'o',  c='r')
plt.plot(XPred_EXP, gp_mean_MD[slicePred_EXP], '-', c='r', label=r"pred. model (post. mean)")
plt.plot(XObs_EXP, gp_mean_MD[sliceObs_EXP], '-', c='r', label=r"pred. model (post. mean)")
upper_MD = gp_mean_MD + 2.0 * np.sqrt(gp_variance_MD)
lower_MD = gp_mean_MD - 2.0 * np.sqrt(gp_variance_MD)
plt.fill_between(XPred_EXP, lower_MD[slicePred_EXP], upper_MD[slicePred_EXP], alpha=0.2, color='tab:red', label=r"PP Credibility interval, $\pm 2 \sigma$")
plt.fill_between(XObs_EXP, lower_MD[sliceObs_EXP], upper_MD[sliceObs_EXP], alpha=0.2, color='tab:red', label=r"PP Credibility interval, $\pm 2 \sigma$")
plt.legend()
plt.savefig(os.path.join(OUTDIR,"plot_GPMD.png"))
plt.close()
#
# plot PPC
gp_mean_PPC = np.mean(gp_samples_PPC, axis=tuple([i for i in range(gp_samples_PPC.ndim-1)]))
gp_variance_PPC = np.squeeze(np.var(np.mean(gp_samples_PPC, axis=0), axis=0))
err = np.sqrt(np.sum((gp_mean_PPC-YObs_EXP)**2))
print("err PCC:", err)
_ = plt.figure()
plt.plot(X_EXP[sliceObs_EXP], YObs_EXP, 'o',  c='black', label=r"training data")
plt.plot(X_EXP[sliceObs_EXP], gp_mean_PPC, 'x',  c='black', label=r"PPC")
err_PPC = 2.0 * np.sqrt(gp_variance_PPC)
plt.errorbar(X_EXP[sliceObs_EXP], gp_mean_PPC, yerr=err_PPC, fmt='x', label=r"PPC")
plt.savefig(os.path.join(OUTDIR,"plot_OBS_PPC.png"))
plt.close()

# Scalar hyperparameters
#
# from posterior (and prior) PDF
#
# length scale
_ = plt.figure()
plt.hist(length_scale_samples_MD.reshape(-1), 30, density=True)
plt.hist(length_scale_samples_MD_prior.reshape(-1), 30, density=True, histtype='step')
plt.xlabel(r"$L_{GP}$")
plt.savefig(os.path.join(OUTDIR,"length_scale_MD.png"))
plt.close()
# MCMC chain
_ = plt.figure()
y = length_scale_samples_MD.reshape(-1)
plt.plot(np.arange(1.,len(y)+1.), y, '-')
plt.xlabel(r"nb samples")
plt.ylabel(r"$L_{GP}$")
plt.savefig(os.path.join(OUTDIR,"length_scale_MD_MCMC.png"))
plt.close()
#
# amplitude
_ = plt.figure()
plt.hist(amplitude_samples_MD.reshape(-1), 30, density=True)
plt.hist(amplitude_samples_MD_prior.reshape(-1), 30, density=True, histtype='step')
plt.xlabel(r"$\sigma^2_{GP}$")
plt.savefig(os.path.join(OUTDIR,"amplitude_MD.png"))
plt.close()
# MCMC chain
_ = plt.figure()
y = amplitude_samples_MD.reshape(-1)
plt.plot(np.arange(1.,len(y)+1.), y, '-')
plt.xlabel(r"nb samples")
plt.ylabel(r"$\sigma^2_{GP}$")
plt.savefig(os.path.join(OUTDIR,"amplitude_MD_MCMC.png"))
plt.close()

# Sim parameters
#
# from posterior (and prior) PDF
#
# llambda
ind_param=0
_ = plt.figure()
plt.hist(theta_samples[...,ind_param].reshape(-1), 30, density=True)
plt.hist(theta_samples_prior[...,ind_param].reshape(-1), 30, density=True, histtype='step')
plt.xlabel(r"$\lambda$")
plt.savefig(os.path.join(OUTDIR,"lambda.png"))
plt.close()
# MCMC chain
_ = plt.figure()
y = theta_samples[...,ind_param].reshape(-1)
plt.plot(np.arange(1.,len(y)+1.), y, '-')
plt.xlabel(r"nb samples")
plt.ylabel(r"$\lambda$")
plt.savefig(os.path.join(OUTDIR,"lambda_MCMC.png"))
plt.close()
#
# beta
ind_param=1
_ = plt.figure()
plt.hist(theta_samples[...,ind_param].reshape(-1), 30, density=True)
plt.hist(theta_samples_prior[...,ind_param].reshape(-1), 30, density=True, histtype='step')
plt.xlabel(r"$\beta$")
plt.savefig(os.path.join(OUTDIR,"beta.png"))
plt.close()
# MCMC chain
_ = plt.figure()
y = theta_samples[...,ind_param].reshape(-1)
plt.plot(np.arange(1.,len(y)+1.), y, '-')
plt.xlabel(r"nb samples")
plt.ylabel(r"$\beta$")
plt.savefig(os.path.join(OUTDIR,"beta_MCMC.png"))
plt.close()

# parameter correlation plots
# theta
_ = plt.figure()
columns=[r"$\lambda$", r"$\beta$"]
#df = pd.DataFrame(theta_samples.reshape(-1,theta_samples.shape[-1]), columns=columns)
#f = sns.pairplot(df, kind="kde")#, corner = True)
#f.map_lower(sns.scatterplot,marker="+")
df1 = pd.DataFrame(theta_samples.reshape(-1,theta_samples.shape[-1]), columns=columns).assign(PDF="posterior")
df0 = pd.DataFrame(theta_samples_prior.reshape(-1,theta_samples_prior.shape[-1]), columns=columns).assign(PDF="prior")
df = df1.append(df0, ignore_index=True)
#f = sns.pairplot(df, kind="kde", hue="PDF", corner = True, plot_kws=dict(common_norm=False, common_grid=True, cut=0, fill=True)); #f.axes[0,0].set_xlim((0,4))
f = sns.jointplot(data=df, x=columns[0], y=columns[1], kind="kde", hue="PDF", plot_kws=dict(common_norm=False, common_grid=True, cut=0), marginal_kws=dict(fill=True))
plt.savefig(os.path.join(OUTDIR,"theta.png"))
plt.close()
# mu
_ = plt.figure()
columns=[r"$L_{GP}$", r"$\sigma^2_{GP}$"]
df1 = pd.DataFrame(mu_samples.reshape(-1,mu_samples.shape[-1]), columns=columns).assign(PDF="posterior")
df0 = pd.DataFrame(mu_samples_prior.reshape(-1,mu_samples_prior.shape[-1]), columns=columns).assign(PDF="prior")
df = df1.append(df0, ignore_index=True)
#f = sns.pairplot(df, kind="kde", hue="PDF", plot_kws=dict(common_norm=False, common_grid=True, cut=0, fill=True))#, corner = True)
#f.map_lower(sns.scatterplot,marker="+")
#f.axes[0,0].set_xlim((0,4))
f = sns.jointplot(data=df, x=columns[0], y=columns[1], kind="kde", hue="PDF", plot_kws=dict(common_norm=False, common_grid=True, cut=0), marginal_kws=dict(fill=True))
#f.plot_joint(sns.kdeplot, plot_kws=dict(common_norm=False, common_grid=True, cut=0))
f.ax_joint.set_xlim((0,4))
f.ax_joint.set_ylim((0,40))
plt.savefig(os.path.join(OUTDIR,"mu.png"))
plt.close()



