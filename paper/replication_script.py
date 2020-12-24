import SILIA
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.stats as sp
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes
from tqdm import tqdm

#Replicate Figures 2 and 3, as well as relevant results.

'''
Generating arrays and dictionaries to represent the time axis, channels,
and reference inputs in the correct format for SILIA. 100 channels,
5 seconds of signal where each timestep is 0.2 ms. Two reference signals with frequencies of 80 and 120Hz.
'''
time = np.arange(0, 5, 1/5000) #seconds
channels = np.arange(0, 100, 1)

frequencies = [80, 120] #Hz
references = []
for freq in frequencies:
    references.append({'time' : time, 'signal' : np.sin(2 * np.pi * freq * time)})


'''
Generating noisy input signals in the correct format for SILIA. Channels 0-20, 40-60 and 80-100
contain only Gaussian noise with a std of 1. Channels 20-40 contains a sin wave oscillating at
80Hz with an amplitude of 1 as well as the Gaussian noise and channels 60-80 contains a sin wave
oscillating at 120Hz with the same amplitude and noise.
'''

def gen_noise(standard_deviation):
    """
    Generates a random sample from a Gaussian distribution with a mean of 0 and 
    specified standard deviation.
    """
    return np.random.normal(0, standard_deviation)

signal = {'time' : time}
sig_vals = []
for t in time:
    row = []
    for channel in channels:
        if (channel >= 0 and channel < 20) or (channel >= 40 and channel < 60) or (channel >= 80 and channel < 100):
            row.append(gen_noise(1))
        elif channel >= 20 and channel < 40:
            row.append(np.sin(2 * np.pi * frequencies[0] * t) + gen_noise(1))
        elif channel >= 60 and channel < 80:
            row.append(np.sin(2 * np.pi * frequencies[1] * t) + gen_noise(1))
    sig_vals.append(row)

signal['signal'] = sig_vals

'''
Performing Lock-in Amplification
'''
LIA = SILIA.Amplifier(0)

out = LIA.amplify(references, signal, num_windows = 4, window_size = 0.33)


'''
Plotting results
'''
intensities = signal['signal']
time = signal['time']
wavelengths = channels
viridis = cm.get_cmap('viridis')
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
fig.set_size_inches([3,3])
x = []
for t in time:
    x += [[t for w in wavelengths]]
x = np.array(x)

y = []
for w in wavelengths:
    y += [[w for t in time]]
y = np.transpose(y)

psm = ax.pcolormesh(y, x ,intensities, cmap=viridis)
cbar = fig.colorbar(psm, ax=ax)
cbar.ax.tick_params(labelsize=15)

ax.set_xlabel("Channel")
ax.set_ylabel("Time (s)")
cbar.ax.set_ylabel("Intensity (arb. units)", labelpad = 15)
ax.set_xlim(0, 100)
for item in (ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label, cbar.ax.yaxis.label]):
    item.set_fontsize(17)
axins = inset_axes(ax, width="50%", height="60%", bbox_to_anchor=(0.2, 0.7, .5, .4),
               bbox_transform=ax.transAxes, loc=2, borderpad=0)
axins.pcolormesh(y, x ,intensities, cmap=viridis)
x1, x2, y1, y2 = 20, 40, 4, 4.05 # specify the limits
axins.set_xlim(x1, x2) # apply the x-limits
axins.set_ylim(y1, y2) # apply the y-limits
plt.yticks([],visible=False)
plt.xticks([],visible=False)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0")
axins.set_title("80Hz", fontdict = {'fontsize' : 'x-large'})

axins2 = inset_axes(ax, width="50%", height="60%", bbox_to_anchor=(0.55, 0.7, .5, .4),
               bbox_transform=ax.transAxes, loc=2, borderpad=0)
axins2.pcolormesh(y, x ,intensities, cmap=viridis)
x1, x2, y1, y2 = 60, 80, 4, 4.05 # specify the limits
axins2.set_xlim(x1, x2) # apply the x-limits
axins2.set_ylim(y1, y2) # apply the y-limits
plt.yticks([],visible=False)
plt.xticks([],visible=False)
mark_inset(ax, axins2, loc1=1, loc2=3, fc="none", ec="0")
axins2.set_title("120Hz", fontdict = {'fontsize' : 'x-large'})
plt.savefig('fig_3_sim_intensity_colormap.png', bbox_inches = 'tight')

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
fig.set_size_inches([3,3])
channels = np.arange(0, 100, 1)
sig_amps = []
noise_amps = []
i = 0
while i < 100:
    if (i<= 20) or (i >= 40 and i<= 60) or (i >= 80):
        sig_amps += [0]
    elif (i > 20 and i < 40) or (i > 60 and i < 80):
        sig_amps += [1/2]
    if (i == 20 or i == 40 or i == 60 or i == 80):
        noise_amps +=[0]
    else:
        noise_amps += [1]
    i += 1
ax.bar(channels, sig_amps, color='r', width = 1, label = "Signal")
ax.bar(channels, noise_amps,bottom=sig_amps, color='b', width = 1, label = "Noise")
for item in (ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
    item.set_fontsize(17)
ax.set_xlabel("Channel")
ax.set_ylabel("Signal:Noise")
plt.legend(bbox_to_anchor = (1, 0.5), loc = "center left", fontsize = 'x-large')
plt.savefig('fig_2_sim_signal_to_noise_bar.svg', bbox_inches = 'tight')



averaged = 0
formats = ['b-', 'r-', 'k-', 'g-', 'm-', 'y-']
fig, ax = plt.subplots(1, 1, figsize=(3,3))
fig.set_size_inches([3,3])

ax.errorbar(channels ,out['reference 1']['magnitudes'], yerr = out['reference 1']['magnitude stds'], capsize = 3, fmt = 'b-', label = "80Hz")
ax.errorbar(channels ,out['reference 2']['magnitudes'], yerr = out['reference 2']['magnitude stds'], capsize = 3, fmt = 'r-', label = "120Hz")
plt.legend(bbox_to_anchor = (1, 0.6), frameon = False, loc = 'center left', title = 'Signal:Noise' + r' $(1:2)$'+'\n\nReference Frequency')
plt.ylabel("Lock-in Magnitude")
plt.xlabel("Channel")

for item in (ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
    item.set_fontsize(17)
plt.savefig('fig_3_two_freq_mag.svg', bbox_inches='tight')


fig, ax = plt.subplots(1, 1, figsize=(3,3))
fig.set_size_inches([3,3])
ax.errorbar(channels ,out['reference 1']['phases'], yerr = out['reference 1']['phase stds'], fmt = 'b-', capsize = 3, label = "80Hz")
ax.set_xlabel("Channel")
ax.set_ylabel("Phase (rads)")
for item in (ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
    item.set_fontsize(17)
plt.ylim(-7, 7)
plt.savefig('fig_3_two_freq_phase_80.svg', bbox_inches='tight')

fig, ax = plt.subplots(1, 1, figsize=(3,3))
fig.set_size_inches([3,3])
ax.errorbar(channels ,out['reference 2']['phases'], yerr = out['reference 2']['phase stds'], capsize = 3, fmt = 'r-', label = "120Hz")
ax.set_xlabel("Channel")
ax.set_ylabel("Phase (rads)")
for item in (ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
    item.set_fontsize(17)
plt.ylim(-7, 7)
plt.savefig('fig_3_two_freq_phase_120.svg', bbox_inches='tight')

'''
Computing some summary statistics on the results
'''

mags1 = out['reference 1']['magnitudes']
mags2 = out['reference 2']['magnitudes']
mags1_err = out['reference 1']['magnitude stds']
mags2_err = out['reference 2']['magnitude stds']
phase1 = out['reference 1']['phases']
phase2 = out['reference 2']['phases']
phase1_err = out['reference 1']['phase stds']
phase2_err = out['reference 2']['phase stds']

mags1_mean, mags1_var = sp.describe(mags1[20:40])[2:4]
mags2_mean, mags2_var = sp.describe(mags2[60:80])[2:4]
mags1_std = np.sqrt(mags1_var)
mags2_std = np.sqrt(mags2_var)

mags1_err = np.mean(mags1_err[20:40])
mags2_err = np.mean(mags2_err[60:80])

print('mags 80Hz mean: ' + str(mags1_mean))
print('mags 120Hz mean: ' + str(mags2_mean))
print('mags 80Hz std: ' + str(mags1_std))
print('mags 120Hz std: ' + str(mags2_std))
print('mags 80Hz predicted err: ' + str(mags1_err))
print('mags 120Hz predicted err: ' + str(mags2_err))
print('mags 80Hz no signal mean: ' + str(np.mean(mags1[0:20]) * 20/100 + np.mean(mags1[40:100]) * 60/100))
print('mags 120Hz no signal mean: ' + str(np.mean(mags2[0:40]) * 40/100 + np.mean(mags1[60:100]) * 40/100))

phase1_mean, phase1_var = sp.describe(phase1[20:40])[2:4]
phase2_mean, phase2_var = sp.describe(phase2[60:80])[2:4]
phase1_std = np.sqrt(phase1_var)
phase2_std = np.sqrt(phase2_var)

phase1_err = np.mean(phase1_err[20:40])
phase2_err = np.mean(phase2_err[60:80])

print('phase 80Hz mean: ' + str(phase1_mean))
print('phase 120Hz mean: ' + str(phase2_mean))
print('phase 80Hz std: ' + str(phase1_std))
print('phase 120Hz std: ' + str(phase2_std))
print('phase 80Hz predicted err: ' + str(phase1_err))
print('phase 120Hz predicted err: ' + str(phase2_err))







#Replicate Figure 4


import timeit
import json

'''
Runtime vs Input Samples
'''

input_runtimes = {}
runtime_types = ['with fit and interp', 'without fit with interp', 'with fit without interp', 'without fit and interp']
num_samples_list = np.round(np.power(1.1, np.arange(50, 122, 1)))
num_channels = 1
num_references = 1
#Number of times to average timing result
num_averages = 50
for runtime_type in runtime_types:
    tmpRuntimes = []
    print('type : ' + runtime_type, flush = True)
    for num_samples in tqdm(num_samples_list, leave = True, position = 0):
        time = np.arange(0, num_samples, 1)
        references = [{'time' : time, 'signal' : np.sin(2 * np.pi * 1/10 * time)}]
        channels = np.arange(0, num_channels, 1)
        sig = []
        for channel in channels:
            sig.append(np.sin(2 * np.pi * 1/10 * time))
        sig = np.array(sig).T
        signal = {'time' : time, 'signal' : sig}
        LIA = SILIA.Amplifier(0, pbar = False)
        runtime = 0
        for i in range(num_averages):
            if runtime_type == 'with fit and interp':
                start = timeit.default_timer()
                out = LIA.amplify(references, signal, interpolate = True)
                end = timeit.default_timer()
            elif runtime_type == 'without fit with interp':
                start = timeit.default_timer()
                out = LIA.amplify(references, signal, fit_ref = False, interpolate = True)
                end = timeit.default_timer()
            elif runtime_type == 'with fit without interp':
                start = timeit.default_timer()
                out = LIA.amplify(references, signal, fit_ref = True, interpolate = False)
                end = timeit.default_timer()
            elif runtime_type == 'without fit and interp':
                start = timeit.default_timer()
                out = LIA.amplify(references, signal, fit_ref = False, interpolate = False)
                end = timeit.default_timer()
            runtime += (end-start)
        tmpRuntimes.append(runtime/num_averages)
    input_runtimes[runtime_type] = tmpRuntimes

with open('input_runtimes.json', 'w') as json_file:
    json.dump(input_runtimes, json_file)


#Plotting the result:
fig, ax = plt.subplots()
for runtime_type in runtime_types:
    ax.plot(num_samples_list, input_runtimes[runtime_type], label = runtime_type)
plt.legend()
plt.xlabel('Input Samples')
plt.ylabel('Runtime (s)')
plt.savefig('fig_4_num_input_samples_runtime.svg', bbox_inches='tight')

'''
Runtime vs Channels
'''

channels_runtimes = {}
runtime_types = ['with fit and interp', 'without fit with interp', 'with fit without interp', 'without fit and interp']
num_channels_list = np.arange(50, 1001, 50)
num_samples = 4096
num_references = 1
#Number of times to average timing result
num_averages = 50
for runtime_type in runtime_types:
    tmpRuntimes = []
    print('type : ' + runtime_type, flush = True)
    for num_channels in tqdm(num_channels_list, leave = True, position = 0):
        time = np.arange(0, num_samples, 1)
        references = [{'time' : time, 'signal' : np.sin(2 * np.pi * 1/10 * time)}]
        channels = np.arange(0, num_channels, 1)
        sig = []
        for channel in channels:
            sig.append(np.sin(2 * np.pi * 1/10 * time))
        sig = np.array(sig).T
        signal = {'time' : time, 'signal' : sig}
        LIA = SILIA.Amplifier(0, pbar = False)
        runtime = 0
        for i in range(num_averages):
            if runtime_type == 'with fit and interp':
                start = timeit.default_timer()
                out = LIA.amplify(references, signal, interpolate = True)
                end = timeit.default_timer()
            elif runtime_type == 'without fit with interp':
                start = timeit.default_timer()
                out = LIA.amplify(references, signal, fit_ref = False, interpolate = True)
                end = timeit.default_timer()
            elif runtime_type == 'with fit without interp':
                start = timeit.default_timer()
                out = LIA.amplify(references, signal, fit_ref = True, interpolate = False)
                end = timeit.default_timer()
            elif runtime_type == 'without fit and interp':
                start = timeit.default_timer()
                out = LIA.amplify(references, signal, fit_ref = False, interpolate = False)
                end = timeit.default_timer()
            runtime += (end-start)
        tmpRuntimes.append(runtime/num_averages)
    channels_runtimes[runtime_type] = tmpRuntimes


with open('channel_runtimes.json', 'w') as json_file:
    json.dump(channels_runtimes, json_file)


#Plotting the result:
fig, ax = plt.subplots()
for runtime_type in runtime_types:
    ax.plot(num_channels_list, channels_runtimes[runtime_type], label = runtime_type)
plt.legend()
plt.xlabel('Channels')
plt.ylabel('Runtime (s)')
plt.savefig('fig_4_num_channels_runtime.svg', bbox_inches='tight')


'''
Runtime vs Number of Frequency References
'''

ref_runtimes = {}
runtime_types = ['with fit and interp', 'without fit with interp', 'with fit without interp', 'without fit and interp']
num_channels = 1
num_samples = 4096
num_references_list = np.arange(1, 11, 1)
#Number of times to average timing result
num_averages = 50
for runtime_type in runtime_types:
    tmpRuntimes = []
    print('type : ' + runtime_type, flush = True)
    for num_references in tqdm(num_references_list, leave = True, position = 0):
        time = np.arange(0, num_samples, 1)
        references = [{'time' : time, 'signal' : np.sin(2 * np.pi * 1/10 * time)}]
        channels = np.arange(0, num_channels, 1)
        sig = []
        for channel in channels:
            sig.append(np.sin(2 * np.pi * 1/10 * time))
        sig = np.array(sig).T
        signal = {'time' : time, 'signal' : sig}
        LIA = SILIA.Amplifier(0, pbar = False)
        runtime = 0
        for i in range(num_averages):
            if runtime_type == 'with fit and interp':
                start = timeit.default_timer()
                out = LIA.amplify(references, signal, interpolate = True)
                end = timeit.default_timer()
            elif runtime_type == 'without fit with interp':
                start = timeit.default_timer()
                out = LIA.amplify(references, signal, fit_ref = False, interpolate = True)
                end = timeit.default_timer()
            elif runtime_type == 'with fit without interp':
                start = timeit.default_timer()
                out = LIA.amplify(references, signal, fit_ref = True, interpolate = False)
                end = timeit.default_timer()
            elif runtime_type == 'without fit and interp':
                start = timeit.default_timer()
                out = LIA.amplify(references, signal, fit_ref = False, interpolate = False)
                end = timeit.default_timer()
            runtime += (end-start)
        tmpRuntimes.append(runtime/num_averages)
    ref_runtimes[runtime_type] = tmpRuntimes


#Plotting the result:
fig, ax = plt.subplots()
for runtime_type in runtime_types:
    ax.plot(num_references_list, ref_runtimes[runtime_type], label = runtime_type)
plt.legend()
plt.xlabel('Frequency References')
plt.ylabel('Runtime (s)')
plt.savefig('fig_4_num_frequencies_runtime.svg', bbox_inches='tight')







#Replicate Figure 5

