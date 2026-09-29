import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns
import pandas as pd

############ LOAD DATA ############
with np.load("output/data_config_EXP.npz", allow_pickle=True) as data:
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

with np.load("output/data_config_SIM.npz", allow_pickle=True) as data:
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
#XObs = np.vstack((XObs0_SIM,XObs_SIM))[...,0]



amplitude_samples_MD = np.load("output/amplitude_samples_MD.npy")
length_scale_samples_MD = np.load("output/length_scale_samples_MD.npy")
gp_samples_MD = np.load("output/gp_samples_MD.npy")
amplitude_samples_SIM = np.load("output/amplitude_samples_SIM.npy")
length_scale_samples_SIM = np.load("output/length_scale_samples_SIM.npy")
theta_samples = np.load("output/theta_samples.npy")
gp_samples_SIM = np.load("output/gp_samples_SIM.npy")
gp_samples_PPC = np.load("output/gp_samples_PPC.npy")
log_accept_ratio = np.load("output/log_accept_ratio.npy")
#p_accept = tf.math.exp(tfp.math.reduce_logmeanexp(tf.minimum(log_accept_ratio, 0.)))
#print(f"Acceptance ratio: {p_accept}")

mu_samples_MD = np.hstack((length_scale_samples_MD, amplitude_samples_MD))
#mu_samples_SIM = np.hstack((length_scale_samples_SIM, amplitude_samples_SIM[...,np.newaxis]))

NbDimT = theta_samples.shape[-1]
NbDimX = 1

amplitude_samples_MD_prior = np.load("output/amplitude_samples_MD_prior.npy")
length_scale_samples_MD_prior = np.load("output/length_scale_samples_MD_prior.npy")
gp_samples_MD_prior = np.load("output/gp_samples_MD_prior.npy")
amplitude_samples_SIM_prior = np.load("output/amplitude_samples_SIM_prior.npy")
length_scale_samples_SIM_prior = np.load("output/length_scale_samples_SIM_prior.npy")
theta_samples_prior = np.load("output/theta_samples_prior.npy")
gp_samples_SIM_prior = np.load("output/gp_samples_SIM_prior.npy")
#mu_samples_MD_prior = np.hstack((length_scale_samples_MD_prior, amplitude_samples_MD_prior))
mu_samples_MD_prior = np.column_stack((length_scale_samples_MD_prior, amplitude_samples_MD_prior))
#mu_samples_SIM_prior = np.hstack((length_scale_samples_SIM_prior, amplitude_samples_SIM_prior))

############ PLOTS ############
# parameters
theta_mean=np.reshape( np.mean(np.mean(theta_samples,axis=0),axis=0), -1)
print('theta_mean =', theta_mean)
amplitude_mean_MD=np.reshape( np.mean(np.mean(amplitude_samples_MD,axis=0),axis=0), -1)
print('amplitude_mean_MD =', amplitude_mean_MD)
length_scale_mean_MD=np.reshape( np.mean(np.mean(length_scale_samples_MD,axis=0),axis=0), -1)
print('length_scale_mean_MD =', length_scale_mean_MD)

# Gaussian process SIM+MD
gp_mean_MD = np.mean(gp_samples_MD, axis=tuple([i for i in range(gp_samples_MD.ndim-1)]))
gp_variance_MD = np.var(gp_samples_MD, axis=tuple([i for i in range(gp_samples_MD.ndim-1)]))
gp_mean_SIM = np.mean(gp_samples_SIM, axis=tuple([i for i in range(gp_samples_SIM.ndim-1)]))
gp_variance_SIM = np.var(gp_samples_SIM, axis=tuple([i for i in range(gp_samples_SIM.ndim-1)]))
gp_mean_PPC = np.mean(gp_samples_PPC, axis=tuple([i for i in range(gp_samples_PPC.ndim-1)]))
gp_variance_PPC = np.var(gp_samples_PPC, axis=tuple([i for i in range(gp_samples_PPC.ndim-1)]))
# following works when XPred_SIM=XPred_EXP
gp_samples_PP = np.append(gp_samples_MD[...,sliceObs_EXP],gp_samples_MD[...,slicePred_EXP],axis = -1) + np.append(gp_samples_SIM[...,sliceObs_EXP],gp_samples_SIM[...,slicePred_SIM],axis = -1)
gp_mean_PP = np.mean(gp_samples_PP, axis=tuple([i for i in range(gp_samples_PP.ndim-1)]))
gp_variance_PP = np.var(gp_samples_PP, axis=tuple([i for i in range(gp_samples_PP.ndim-1)]))
#
_ = plt.figure()
plt.plot(XObs_EXP, YObs_EXP, 'o',  c='r')
plt.plot(XPred_EXP, gp_mean_SIM[slicePred_SIM], '-.', c='r', label=r"sim. model (post. mean)")
plt.plot(XPred_EXP, gp_mean_PP[slicePred_EXP], '-', c='r', label=r"pred. model (post. mean)")
plt.plot(XObs_EXP, gp_mean_PP[sliceObs_EXP], '-', c='r', label=r"pred. model (post. mean)")
upper = gp_mean_PP + 2.0 * np.sqrt(gp_variance_PP)
lower = gp_mean_PP - 2.0 * np.sqrt(gp_variance_PP)
plt.fill_between(XPred_EXP, lower[slicePred_EXP], upper[slicePred_EXP], alpha=0.2, color='tab:red', label=r"PP Credibility interval, $\pm 2 \sigma$")
plt.fill_between(XObs_EXP, lower[sliceObs_EXP], upper[sliceObs_EXP], alpha=0.2, color='tab:red', label=r"PP Credibility interval, $\pm 2 \sigma$")
err_PPC = 2.0 * np.sqrt(gp_variance_PPC[sliceObs_EXP])
plt.errorbar(XObs_EXP, gp_mean_PPC[sliceObs_EXP], yerr=err_PPC, fmt='x', label=r"PPC")
plt.legend()
plt.savefig("output/plot_GPMAP.png")
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
plt.savefig("output/length_scale_MD.png")
plt.clf()
# MCMC chain
_ = plt.figure()
y = length_scale_samples_MD.reshape(-1)
plt.plot(np.arange(1.,len(y)+1.), y, '-')
plt.xlabel(r"nb samples")
plt.ylabel(r"$L_{GP}$")
plt.savefig("output/length_scale_MD_MCMC.png")
plt.clf()
#
# amplitude MD
_ = plt.figure()
plt.hist(amplitude_samples_MD.reshape(-1), 30, density=True)
plt.hist(amplitude_samples_MD_prior.reshape(-1), 30, density=True, histtype='step')
plt.xlabel(r"$\sigma^2_{GP}$")
plt.savefig("output/amplitude_MD.png")
plt.clf()
# MCMC chain
_ = plt.figure()
y = amplitude_samples_MD.reshape(-1)
plt.plot(np.arange(1.,len(y)+1.), y, '-')
plt.xlabel(r"nb samples")
plt.ylabel(r"$\sigma^2_{GP}$")
plt.savefig("output/amplitude_MD_MCMC.png")
plt.clf()
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
f = sns.jointplot(data=df, x=columns[0], y=columns[1], marker="+", hue="PDF")
f.plot_joint(sns.kdeplot)
f.ax_joint.set_xlim((0,4))
f.ax_joint.set_ylim((0,40))
plt.savefig("output/mu_MD.png")
plt.clf()



#########
# SIM
#########
# length scale SIM
y0 = length_scale_samples_SIM[...,0].reshape(-1)
y1 = length_scale_samples_SIM[...,1].reshape(-1)
y0p = length_scale_samples_SIM_prior[...,0].reshape(-1)
y1p = length_scale_samples_SIM_prior[...,1].reshape(-1)

_ = plt.figure()
plt.hist(y0, 30, density=True)
plt.hist(y0p, 30, density=True, histtype='step')
plt.xlabel(r"$L_{GP}$")
plt.savefig("output/length_scale_SIM.png")
plt.clf()
# MCMC chain
_ = plt.figure()
plt.plot(np.arange(1.,len(y0)+1.), y0, '-')
plt.plot(np.arange(1.,len(y1)+1.), y1, '-')
plt.xlabel(r"nb samples")
plt.ylabel(r"$L_{GP}$")
plt.savefig("output/length_scale_SIM_MCMC.png")
plt.clf()
#
# amplitude SIM
_ = plt.figure()
plt.hist(amplitude_samples_SIM.reshape(-1), 30, density=True)
plt.hist(amplitude_samples_SIM_prior.reshape(-1), 30, density=True, histtype='step')
plt.xlabel(r"$\sigma^2_{GP}$")
plt.savefig("output/amplitude_SIM.png")
plt.clf()
# MCMC chain
_ = plt.figure()
y = amplitude_samples_SIM.reshape(-1)
plt.plot(np.arange(1.,len(y)+1.), y, '-')
plt.xlabel(r"nb samples")
plt.ylabel(r"$\sigma^2_{GP}$")
plt.savefig("output/amplitude_SIM_MCMC.png")
plt.clf()
#
# material parameters
#
# llambda
ind_param=0
_ = plt.figure()
plt.hist(theta_samples[...,ind_param].reshape(-1), 30, density=True)
plt.hist(theta_samples_prior[...,ind_param].reshape(-1), 30, density=True, histtype='step')
plt.xlabel(r"$\lambda$")
plt.savefig("output/lambda.png")
plt.clf()
# MCMC chain
_ = plt.figure()
y = theta_samples[...,ind_param].reshape(-1)
plt.plot(np.arange(1.,len(y)+1.), y, '-')
plt.xlabel(r"nb samples")
plt.ylabel(r"$\lambda$")
plt.savefig("output/lambda_MCMC.png")
plt.clf()
