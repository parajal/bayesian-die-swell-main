import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
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

############ LOAD DATA ############
INDIR = INDIRS[0]
with np.load(os.path.join(INDIR,"data_config_EXP.npz"), allow_pickle=True) as data:
    try:
        YObs_EXP = data['arr_0']
        X_EXP = data['arr_1']
        sliceObs_EXP = data['arr_2'].item()
        slicePred_EXP = data['arr_3'].item()
    except:
        YObs_EXP = data['arr_0'][0]
        X_EXP = data['arr_0'][1]
        sliceObs_EXP = data['arr_0'][2].item()
        slicePred_EXP = data['arr_0'][3].item()

with np.load(os.path.join(INDIR,"data_config_SIM.npz"), allow_pickle=True) as data:
    try:
        YObs_SIM = data['arr_0']
        X_SIM = data['arr_1']
        sliceObs_SIM = data['arr_2'].item()
        slicePred_SIM = data['arr_3'].item()
    except:
        YObs_SIM = data['arr_0'][0]
        X_SIM = data['arr_0'][1]
        sliceObs_SIM = data['arr_0'][2].item()
        slicePred_SIM = data['arr_0'][3].item()

XObs_EXP = X_EXP[sliceObs_EXP]
XPred_EXP = X_EXP[slicePred_EXP]
XObs_SIM = X_SIM[sliceObs_SIM]
XPred_SIM = X_SIM[slicePred_SIM]
XObs_SIM_x = XObs_SIM[...,0]
XPred_SIM_x = XPred_SIM[...,0]
#XObs = np.vstack((XObs0_SIM,XObs_SIM))[...,0]



amplitude_samples_MD = np.load(os.path.join(INDIR,"amplitude_samples_MD.npy"))
length_scale_samples_MD = np.load(os.path.join(INDIR,"length_scale_samples_MD.npy"))
gp_samples_MD = np.load(os.path.join(INDIR,"gp_samples_MD.npy"))
amplitude_samples_SIM = np.load(os.path.join(INDIR,"amplitude_samples_SIM.npy"))
length_scale_samples_SIM = np.load(os.path.join(INDIR,"length_scale_samples_SIM.npy"))
theta_samples = np.load(os.path.join(INDIR,"theta_samples.npy"))
gp_samples_SIM = np.load(os.path.join(INDIR,"gp_samples_SIM.npy"))
#gp_samples_PPC = np.load(os.path.join(INDIR,"gp_samples_PPC.npy"))
log_accept_ratio = np.load(os.path.join(INDIR,"log_accept_ratio.npy"))
#p_accept = tf.math.exp(tfp.math.reduce_logmeanexp(tf.minimum(log_accept_ratio, 0.)))
#print(f"Acceptance ratio: {p_accept}")

mu_samples_MD = np.column_stack((length_scale_samples_MD, amplitude_samples_MD))
mu_samples_SIM = np.column_stack((np.squeeze(length_scale_samples_SIM), amplitude_samples_SIM)) # np.squeeze only if length_scale_samples_SIM.shape[-1]>1

NbDimT = theta_samples.shape[-1]
NbDimX = 1

#amplitude_samples_MD_prior = np.load(os.path.join(INDIR,"amplitude_samples_MD_prior.npy"))
#length_scale_samples_MD_prior = np.load(os.path.join(INDIR,"length_scale_samples_MD_prior.npy"))
#gp_samples_MD_prior = np.load(os.path.join(INDIR,"gp_samples_MD_prior.npy"))
#amplitude_samples_SIM_prior = np.load(os.path.join(INDIR,"amplitude_samples_SIM_prior.npy"))
#length_scale_samples_SIM_prior = np.load(os.path.join(INDIR,"length_scale_samples_SIM_prior.npy"))
#theta_samples_prior = np.load(os.path.join(INDIR,"theta_samples_prior.npy"))
#gp_samples_SIM_prior = np.load(os.path.join(INDIR,"gp_samples_SIM_prior.npy"))
##mu_samples_MD_prior = np.hstack((length_scale_samples_MD_prior, amplitude_samples_MD_prior))
#mu_samples_MD_prior = np.column_stack((length_scale_samples_MD_prior, amplitude_samples_MD_prior))
#mu_samples_SIM_prior = np.column_stack((np.squeeze(length_scale_samples_SIM_prior), amplitude_samples_SIM_prior)) # np.squeeze only if length_scale_samples_SIM.shape[-1]>1

# Prior
nsamples_prior = len(theta_samples)  # needed until fixed PDE normalization in joint-/pairplot
samples_prior = gp_model.sample(nsamples_prior)
amplitude_samples_MD_prior = samples_prior[0].numpy()
length_scale_samples_MD_prior = samples_prior[1].numpy()
gp_samples_MD_prior = samples_prior[2].numpy()
amplitude_samples_SIM_prior = samples_prior[3].numpy()
length_scale_samples_SIM_prior = samples_prior[4].numpy()
theta_samples_prior = samples_prior[5].numpy()
gp_samples_SIM_prior = samples_prior[6].numpy()
mu_samples_MD_prior = np.column_stack((length_scale_samples_MD_prior, amplitude_samples_MD_prior))
mu_samples_SIM_prior = np.column_stack((np.squeeze(length_scale_samples_SIM_prior), amplitude_samples_SIM_prior)) # np.squeeze only if length_scale_samples_SIM.shape[-1]>1

# PPC
GP_PPC = gp_model.experimental_pin(
    amplitude_MD=amplitude_samples_MD,
    length_scale_MD=length_scale_samples_MD,
    GP_MD=gp_samples_MD,
    amplitude_SIM=amplitude_samples_SIM,
    scale_diag_SIM=length_scale_samples_SIM,
    theta=theta_samples,
    GP_SIM=gp_samples_SIM)

nsamples_per_mcmc_sample = 100
gp_samples_PPC = GP_PPC.sample_unpinned(nsamples_per_mcmc_sample)[0].numpy()


############ PLOTS ############
# parameters
theta_mean=np.reshape( np.mean(np.mean(theta_samples,axis=0),axis=0), -1)
print('theta_mean =', theta_mean)
amplitude_mean_MD=np.reshape( np.mean(np.mean(amplitude_samples_MD,axis=0),axis=0), -1)
print('amplitude_mean_MD =', amplitude_mean_MD)
length_scale_mean_MD=np.reshape( np.mean(np.mean(length_scale_samples_MD,axis=0),axis=0), -1)
print('length_scale_mean_MD =', length_scale_mean_MD)
#
# Gaussian process SIM+MD
#
gp_mean_MD = np.mean(gp_samples_MD, axis=tuple([i for i in range(gp_samples_MD.ndim-1)]))
#gp_variance_MD = np.var(gp_samples_MD, axis=tuple([i for i in range(gp_samples_MD.ndim-1)]))
gp_variance_MD = np.squeeze(np.var(gp_samples_MD, axis=0))
gp_mean_SIM = np.mean(gp_samples_SIM, axis=tuple([i for i in range(gp_samples_SIM.ndim-1)]))
#gp_variance_SIM = np.var(gp_samples_SIM, axis=tuple([i for i in range(gp_samples_SIM.ndim-1)]))
gp_variance_SIM = np.squeeze(np.var(gp_samples_SIM, axis=0))
gp_mean_PPC = np.mean(gp_samples_PPC, axis=tuple([i for i in range(gp_samples_PPC.ndim-1)]))
#gp_variance_PPC = np.var(gp_samples_PPC, axis=tuple([i for i in range(gp_samples_PPC.ndim-1)]))
gp_variance_PPC = np.squeeze(np.var(np.mean(gp_samples_PPC, axis=0), axis=0)) # here, np.var(np.mean(...)), as multiple samples in axis=0 per posterior sample
# following works when XPred_SIM=XPred_EXP
gp_samples_PP = np.append(gp_samples_MD[...,sliceObs_EXP],gp_samples_MD[...,slicePred_EXP],axis = -1) + np.append(gp_samples_SIM[...,sliceObs_EXP],gp_samples_SIM[...,slicePred_SIM],axis = -1)
gp_mean_PP = np.mean(gp_samples_PP, axis=tuple([i for i in range(gp_samples_PP.ndim-1)]))
gp_variance_PP = np.squeeze(np.var(gp_samples_PP, axis=0))
#
_ = plt.figure()
plt.plot(XObs_EXP, YObs_EXP, 'o',  c='r')
plt.plot(XPred_EXP, gp_mean_SIM[slicePred_SIM], '-.', c='r', label=r"sim. model (post. mean)")
plt.plot(XPred_EXP, gp_mean_PP[slicePred_EXP], '-', c='r', label=r"pred. model (post. mean)")
#plt.plot(XObs_EXP, gp_mean_PP[sliceObs_EXP], '-', c='r', label=r"pred. model (post. mean)")
upper = gp_mean_PP + 2.0 * np.sqrt(gp_variance_PP)
lower = gp_mean_PP - 2.0 * np.sqrt(gp_variance_PP)
plt.fill_between(XPred_EXP, lower[slicePred_EXP], upper[slicePred_EXP], alpha=0.2, color='tab:red', label=r"PP Credibility interval, $\pm 2 \sigma$")
#plt.fill_between(XObs_EXP, lower[sliceObs_EXP], upper[sliceObs_EXP], alpha=0.2, color='tab:red', label=r"PP Credibility interval, $\pm 2 \sigma$")
err_PPC = 2.0 * np.sqrt(gp_variance_PPC[sliceObs_EXP])
plt.errorbar(XObs_EXP, gp_mean_PPC[sliceObs_EXP], yerr=err_PPC, fmt='x', label=r"PPC")
plt.legend()
plt.savefig(os.path.join(OUTDIR,"plot_GPMAP.png"))
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
# Gaussian process SIM
_ = plt.figure()
plt.plot(XObs_EXP, YObs_EXP, 'o',  c='r', label=r"exp. obs.")
plt.plot(XObs_SIM_x, YObs_SIM, 'o',  c='r', alpha=0.2, label=r"sim. obs. $f(\theta)$")
plt.plot(XPred_SIM_x, gp_mean_SIM[slicePred_SIM], '-', c='r', label=r"pred. model (post. mean)")
#plt.plot(XObs_SIM_x, gp_mean_SIM[sliceObs_SIM], '-', c='r', label=r"pred. model (post. mean)")
upper_SIM = gp_mean_SIM + 2.0 * np.sqrt(gp_variance_SIM)
lower_SIM = gp_mean_SIM - 2.0 * np.sqrt(gp_variance_SIM)
plt.fill_between(XPred_SIM_x, lower_SIM[slicePred_SIM], upper_SIM[slicePred_SIM], alpha=0.2, color='tab:red', label=r"PP Credibility interval, $\pm 2 \sigma$")
#plt.fill_between(XObs_SIM_x, lower_SIM[sliceObs_SIM], upper_SIM[sliceObs_SIM], alpha=0.2, color='tab:red', label=r"PP Credibility interval, $\pm 2 \sigma$")
plt.legend()
plt.savefig(os.path.join(OUTDIR,"plot_GPSIM.png"))
plt.close()
#
# plot PPC
gp_mean_PPC = np.mean(gp_samples_PPC, axis=tuple([i for i in range(gp_samples_PPC.ndim-1)]))
gp_variance_PPC = np.squeeze(np.var(np.mean(gp_samples_PPC, axis=0), axis=0))
#err = np.sqrt(np.sum((gp_mean_PPC-YObs_EXP)**2))
#print("err PCC:", err)
# on YObs_EXP
_ = plt.figure()
plt.plot(XObs_EXP, YObs_EXP, 'o',  c='black', label=r"training data")
plt.plot(XObs_EXP, gp_mean_PPC[sliceObs_EXP], 'x',  c='black', label=r"PPC")
#err_PPC = 2.0 * np.sqrt(gp_variance_PPC[sliceObs_EXP])
#plt.errorbar(XObs_EXP, gp_mean_PPC[sliceObs_EXP], yerr=err_PPC, fmt='x', c='black', label=r"PPC error")
plt.legend()
plt.savefig(os.path.join(OUTDIR,"plot_PPC_EXP.png"))
plt.clf()
# on YObs_SIM
_ = plt.figure()
plt.plot(XObs_SIM, YObs_SIM, 'o',  c='black', label=r"training data")
plt.plot(XObs_SIM, gp_mean_PPC[sliceObs_SIM], 'x',  c='black', label=r"PPC")
#err_PPC = 2.0 * np.sqrt(gp_variance_PPC[sliceObs_SIM])
#plt.errorbar(XObs_SIM, gp_mean_PPC[sliceObs_SIM], yerr=err_PPC, fmt='x', c='black', label=r"PPC error")
plt.legend()
plt.savefig(os.path.join(OUTDIR,"plot_PPC_SIM.png"))
plt.clf()

# Parameter plots
# from posterior (and prior) PDF

####
# MD
####
# length scale MD
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
# amplitude MD
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
#
# parameter correlation plots
#
# mu_MD
_ = plt.figure()
columns=[r"$L_{GP}$", r"$\sigma^2_{GP}$"]
df1 = pd.DataFrame(mu_samples_MD.reshape(-1,mu_samples_MD.shape[-1]), columns=columns).assign(PDF="posterior")
df0 = pd.DataFrame(mu_samples_MD_prior.reshape(-1,mu_samples_MD_prior.shape[-1]), columns=columns).assign(PDF="prior")
df = df1.append(df0, ignore_index=True)
#f = sns.pairplot(df, kind="kde", hue="PDF")#, corner = True)
#f.map_lower(sns.scatterplot,marker="+")
#f.axes[0,0].set_xlim((0,4))
f = sns.jointplot(data=df, x=columns[0], y=columns[1], marker=".", s=0.1, hue="PDF")
f.plot_joint(sns.kdeplot)
f.ax_joint.set_xlim((0,4))
f.ax_joint.set_ylim((0,40))
plt.savefig(os.path.join(OUTDIR,"mu_MD.png"))
plt.close()



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

_ = plt.figure()
plt.hist(y0, 30, density=True)
plt.hist(y0p, 30, density=True, histtype='step')
plt.xlabel(r"$L_{GP}$")
plt.savefig(os.path.join(OUTDIR,"length_scale_SIM.png"))
plt.close()
# MCMC chain
_ = plt.figure()
plt.plot(np.arange(1.,len(y0)+1.), y0, '-')
plt.plot(np.arange(1.,len(y1)+1.), y1, '-')
plt.plot(np.arange(1.,len(y2)+1.), y2, '-')
plt.xlabel(r"nb samples")
plt.ylabel(r"$L_{GP}$")
plt.savefig(os.path.join(OUTDIR,"length_scale_SIM_MCMC.png"))
plt.close()
#
# amplitude SIM
_ = plt.figure()
plt.hist(amplitude_samples_SIM.reshape(-1), 30, density=True)
plt.hist(amplitude_samples_SIM_prior.reshape(-1), 30, density=True, histtype='step')
plt.xlabel(r"$\sigma^2_{GP}$")
plt.savefig(os.path.join(OUTDIR,"amplitude_SIM.png"))
plt.close()
# MCMC chain
_ = plt.figure()
y = amplitude_samples_SIM.reshape(-1)
plt.plot(np.arange(1.,len(y)+1.), y, '-')
plt.xlabel(r"nb samples")
plt.ylabel(r"$\sigma^2_{GP}$")
plt.savefig(os.path.join(OUTDIR,"amplitude_SIM_MCMC.png"))
plt.close()
#
# material parameters
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
#
# parameter correlation plots
#
# theta
_ = plt.figure()
columns=[r"$\lambda$", r"$\beta$"]
#df = pd.DataFrame(theta_samples.reshape(-1,theta_samples.shape[-1]), columns=columns)
#f = sns.pairplot(df, kind="kde")#, corner = True)
#f.map_lower(sns.scatterplot,marker="+")
df1 = pd.DataFrame(theta_samples.reshape(-1,NbDimT), columns=columns).assign(PDF="posterior")
df0 = pd.DataFrame(theta_samples_prior.reshape(-1,NbDimT), columns=columns).assign(PDF="prior")
df = df1.append(df0, ignore_index=True)
#f = sns.pairplot(df, kind="kde", hue="PDF", corner = True); #f.axes[0,0].set_xlim((0,4))
#f = sns.jointplot(data=df, x=columns[0], y=columns[1], marker="+", hue="PDF")
f = sns.jointplot(data=df, x=columns[0], y=columns[1], marker=".", s=0.1, hue="PDF")
f.plot_joint(sns.kdeplot)
plt.savefig(os.path.join(OUTDIR,"theta.png"))
plt.close()
#
# mu_SIM
_ = plt.figure()
columns=[r"$L_{GP,x}$", r"$L_{GP,\lambda}$", r"$L_{GP,\beta}$", r"$\sigma^2_{GP}$"]
df1 = pd.DataFrame(mu_samples_SIM.reshape(-1,mu_samples_SIM.shape[-1]), columns=columns).assign(PDF="posterior")
df0 = pd.DataFrame(mu_samples_SIM_prior.reshape(-1,mu_samples_SIM_prior.shape[-1]), columns=columns).assign(PDF="prior")
df = df1.append(df0, ignore_index=True)
f = sns.pairplot(df, kind="kde", hue="PDF", corner = True)
f.map_lower(sns.scatterplot,marker=".", s=0.1)
f.axes[0,0].set_xlim((0,20)) # x axis of row 0 col. 0
f.axes[1,1].set_xlim((0,20)) # x axis of row 1 col. 1
f.axes[1,1].set_ylim((0,10)) # y axis of row 1 col. 1
f.axes[2,2].set_xlim((0,20)) # x axis of row 2 col. 2
f.axes[2,2].set_ylim((0,10)) # y axis of row 2 col. 2
f.axes[3,3].set_xlim((0,20)) # x axis of row 3 col. 3
f.axes[3,3].set_ylim((0,10)) # y axis of row 3 col. 3
#f = sns.jointplot(data=df, x=columns[0], y=columns[1], marker="+", hue="PDF")
#f.plot_joint(sns.kdeplot)
#f.ax_joint.set_xlim((0,4))
#f.ax_joint.set_ylim((0,40))
plt.savefig(os.path.join(OUTDIR,"mu_SIM.png"))
plt.close()


#
# response surface
#
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
    xs_obs = XObs_EXP.flatten()
    ys_obs = np.ones_like(xs_obs)*theta_mean[...,ind_param-1]
    zs_obs = YObs_EXP.flatten()
    ax.scatter(xs_obs, ys_obs, zs_obs, marker='o', alpha=1, s=2, c='g', label=r"exp. obs.")
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
    
    ax.set_zlabel(r'GP SIM samples from the posterior')
    plt.legend()
    plt.savefig(os.path.join(OUTDIR,f"plot_GPSIM_surf_{ind_param}.png"), dpi=200)
    print(f"saved figure to file {OUTDIR}/plot_GPSIM_surf_{ind_param}.png")
    plt.close()
