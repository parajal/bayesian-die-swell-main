import os
import matplotlib.pyplot as plt
import numpy as np

output="output"

SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIG_SIZE = 16
plt.rc('font', size=BIG_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIG_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIG_SIZE)  # fontsize of the figure title

############ LOAD DATA ############
with np.load(os.path.join(output,"data_config.npz"), allow_pickle=True) as data:
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

XObs_EXP = X_EXP[sliceObs_EXP]
XPred_EXP = X_EXP[slicePred_EXP]
ZObs = YObs_EXP

amplitude_samples_MD = np.load(os.path.join(output,"amplitude_samples_MD.npy"))
length_scale_samples_MD = np.load(os.path.join(output,"length_scale_samples_MD.npy"))
gp_samples_MD = np.load(os.path.join(output,"gp_samples_MD.npy"))
theta_samples = np.load(os.path.join(output,"theta_samples.npy"))
gp_samples_PPC = np.load("output/gp_samples_PPC.npy")

log_accept_ratio = np.load(os.path.join(output,"log_accept_ratio.npy"))
#p_accept = tf.math.exp(tfp.math.reduce_logmeanexp(tf.minimum(log_accept_ratio, 0.)))
#print(f"Acceptance ratio: {p_accept}")


############ SIMULATION & EXPERIMENTAL MODEL ############
#from koh import f_sim, f_exp

# simulator
def f_sim(x, t):
    beta = t[..., 0] #*0.
    return beta*x

# synthetic experiments
# example form paper [Brynjarsdottir & O'Hagen., 2014]
def f_exp(x):
    beta = .25
    a = 3.
    return beta*x/(1+x/a)

#YPred_EXP = f_exp(X_EXP[slicePred_EXP])

observation_noise_variance = 0.001
np.random.seed(42)
YPred_EXP = np.random.normal(
    size=(len(XPred_EXP),),
    loc=f_exp(XPred_EXP),
    scale=np.sqrt(observation_noise_variance))
#print(X_EXP[slicePred_EXP], YPred_EXP)

############ PLOTS ############
# parameters
theta_mean=np.reshape( np.mean(np.mean(theta_samples,axis=0),axis=0), -1)
theta_variance = np.reshape( np.mean(np.var(theta_samples, axis=0), axis=0), -1)
print('theta_mean =', theta_mean, 'theta_variance =', theta_variance)
amplitude_mean_MD=np.reshape( np.mean(np.mean(amplitude_samples_MD,axis=0),axis=0), -1)
print('amplitude_mean_MD =', amplitude_mean_MD)
length_scale_mean_MD=np.reshape( np.mean(np.mean(length_scale_samples_MD,axis=0),axis=0), -1)
print('length_scale_mean_MD =', length_scale_mean_MD)


### DATA POINTS
# Gaussian process ML
_ = plt.figure()
plt.plot(X_EXP[sliceObs_EXP], YObs_EXP, 'o',  c='black', label=r"training data")
plt.ylabel(r"Die Swell")
plt.xlabel(r"Shear Rate")

plt.xlim((0., 6.5))   # set the xlim to left, right
plt.ylim((0., .6))   # set the xlim to left, right
plt.legend()
plt.savefig(os.path.join(output,"plot_OBS_train.png"))
plt.plot(X_EXP[slicePred_EXP], YPred_EXP, 'x',  c='black', label=r"test data")
plt.legend()
plt.savefig(os.path.join(output,"plot_OBS_test.png"))
plt.clf()

### MODEL PLOTS

# Gaussian process SIM+MD
_ = plt.figure()
gp_mean_MD = np.reshape( np.mean(np.mean(gp_samples_MD, axis=0), axis=0), -1)
gp_variance_MD = np.reshape( np.mean(np.var(gp_samples_MD, axis=0), axis=0), -1)
gp_mean_PP = np.mean(gp_samples_MD.reshape(f_sim(X_EXP, theta_samples).shape) + f_sim(X_EXP, theta_samples),axis=0)
gp_variance_PP = np.var(gp_samples_MD.reshape(f_sim(X_EXP, theta_samples).shape) + f_sim(X_EXP, theta_samples),axis=0)

plt.plot(X_EXP[sliceObs_EXP], YObs_EXP, 'o',  c='black', label=r"training data")
#plt.plot(X_EXP[slicePred_EXP], YPred_EXP, 'x',  c='black', label=r"test data")
plt.plot(X_EXP[slicePred_EXP], f_sim(X_EXP[slicePred_EXP], theta_mean), '-.', color='tab:red', label=r"SIM")
plt.plot(X_EXP[slicePred_EXP], gp_mean_MD[slicePred_EXP] + f_sim(X_EXP[slicePred_EXP], theta_mean), '-', color='tab:green', label=r"SIM+ML")

upper_SIM = f_sim(X_EXP[slicePred_EXP], theta_mean + 2.0 * np.sqrt(theta_variance))
lower_SIM = f_sim(X_EXP[slicePred_EXP], theta_mean - 2.0 * np.sqrt(theta_variance))
#plt.fill_between(X_EXP[slicePred_EXP], lower_SIM, upper_SIM,
#                 alpha=0.2, color='tab:red', label=r"Credibility interval, $\pm 2 \sigma_{\mathrm{SIM}}$")

upper = gp_mean_PP + 2.0 * np.sqrt(gp_variance_PP)
lower = gp_mean_PP - 2.0 * np.sqrt(gp_variance_PP)
#plt.plot(X_EXP[slicePred_EXP], lower, c='r', alpha=0.15)
plt.fill_between(X_EXP[slicePred_EXP], lower[slicePred_EXP], upper[slicePred_EXP],
                 alpha=0.2, color='tab:green', label=r"Credibility interval, $\pm 2 \sigma$")

plt.ylabel(r"Die Swell")
plt.xlabel(r"Shear Rate")
#plt.legend()
plt.savefig(os.path.join(output,"plot_GPMAP_train.png"))
plt.plot(X_EXP[slicePred_EXP], YPred_EXP, 'x',  c='black', label=r"test data")
plt.savefig(os.path.join(output,"plot_GPMAP_test.png"))
plt.clf()



# calibrated SIM
_ = plt.figure()
plt.plot(X_EXP[sliceObs_EXP], YObs_EXP, 'o',  c='black', label=r"training data")
#plt.plot(X_EXP[slicePred_EXP], YPred_EXP, 'x',  c='black', label=r"test data")
plt.plot(X_EXP[slicePred_EXP], f_sim(X_EXP[slicePred_EXP], theta_mean), '-.', color='tab:red', label=r"SIM")

#plt.fill_between(X_EXP[slicePred_EXP], lower_SIM, upper_SIM,
#                 alpha=0.2, color='tab:red', label=r"Credibility interval, $\pm 2 \sigma_{\mathrm{SIM}}$")

plt.ylabel(r"Die Swell")
plt.xlabel(r"Shear Rate")
#plt.legend()
plt.savefig(os.path.join(output,"plot_GPSIM_train.png"))
plt.plot(X_EXP[slicePred_EXP], YPred_EXP, 'x',  c='black', label=r"test data")
plt.savefig(os.path.join(output,"plot_GPSIM_test.png"))
plt.clf()


# Gaussian process ML
_ = plt.figure()
plt.plot(X_EXP[sliceObs_EXP], YObs_EXP, 'o',  c='black', label=r"training data")
#plt.plot(X_EXP[slicePred_EXP], YPred_EXP, 'x',  c='black', label=r"test data")
plt.plot(X_EXP[slicePred_EXP], gp_mean_MD[slicePred_EXP], '-', color='tab:blue', label=r"ML")

upper_MD = gp_mean_MD[slicePred_EXP]  + 2.0 * np.sqrt(gp_variance_MD[slicePred_EXP])
lower_MD = gp_mean_MD[slicePred_EXP]  - 2.0 * np.sqrt(gp_variance_MD[slicePred_EXP])
plt.fill_between(X_EXP[slicePred_EXP], lower_MD, upper_MD,
                 alpha=0.2, color='tab:blue', label=r"Credibility interval, $\pm 2 \sigma_{\mathrm{ML}}$")

plt.ylabel(r"Die Swell")
plt.xlabel(r"Shear Rate")
#plt.legend()
plt.savefig(os.path.join(output,"plot_GPMD_train.png"))
plt.plot(X_EXP[slicePred_EXP], YPred_EXP, 'x',  c='black', label=r"test data")
plt.savefig(os.path.join(output,"plot_GPMD_test.png"))
plt.clf()



# Scalar hyperparameters
_ = plt.figure()
plt.hist(length_scale_samples_MD.reshape(-1), 30, density=True)
plt.xlabel(r"$L_{GP}$")
plt.savefig(os.path.join(output,"length_scale_MD.png"))
plt.clf()

_ = plt.figure()
plt.hist(amplitude_samples_MD.reshape(-1), 30, density=True)
plt.xlabel(r"$\sigma^2_{GP}$")
plt.savefig(os.path.join(output,"amplitude_MD.png"))
plt.clf()

# Sim parameters
#
# from posterior PDF
# llambda
ind_param=0
_ = plt.figure()
plt.hist(theta_samples[...,ind_param].reshape(-1), 30, density=True)
plt.xlabel(r"$\theta$")
plt.savefig(os.path.join(output,"theta.png"))
plt.clf()
# MCMC chain
_ = plt.figure()
y = theta_samples[...,ind_param].reshape(-1)
plt.plot(np.arange(1.,len(y)+1.), y, '-')
plt.xlabel(r"nb samples")
plt.ylabel(r"$\theta$")
plt.savefig(os.path.join(output,"theta_MCMC.png"))
plt.clf()
