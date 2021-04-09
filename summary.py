import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os


digit = "10"
algo = "RXCSi"
run_start = 1
run_end = 30
epochs = 100
base_path = "../remote/output/" + algo +"/output-" + digit + "-digit/"
output_folder = "summary/"

if not os.path.exists(base_path + output_folder):
    os.makedirs(base_path + output_folder)


def save_max_rows():
    training_max = []
    validation_max = []
    for i in range(run_start, run_end+1):
        run = str(i).zfill(2)
        test_performance = np.loadtxt(base_path + digit + "-digits-" + run + "/test_performance.txt")
        t_max = test_performance[np.argmax(test_performance[:, 1]), :]
        training_max.append(t_max)
        v_max = test_performance[np.argmax(test_performance[:, 4]), :]
        validation_max.append(v_max)

    fmt = '%d %7f %7f %d %7f %7f %d'
    np.savetxt(base_path + output_folder+algo+'-'+digit+'-'+str(run_start).zfill(2)+'-'+str(run_end).zfill(2)+'-training_max.txt', training_max, fmt=fmt)
    np.savetxt(base_path + output_folder+algo+'-'+digit+'-'+str(run_start).zfill(2)+'-'+str(run_end).zfill(2)+'-validation_max.txt', validation_max, fmt=fmt)


def sav_averages():
    collect = []
    for i in range(run_start, run_end+1):
        run = str(i).zfill(2)
        test_performance = np.loadtxt(base_path + digit + "-digits-" + run + "/test_performance.txt")
        collect.append(test_performance)

    output = np.empty([0, 7])
    output_sd = np.empty([0, 7])
    for i in range(epochs):
        avg = np.empty([0, 7])
        for j in range(run_end - run_start + 1):
            avg = np.concatenate((avg, np.reshape((collect[j])[i, :], (1, -1))), axis=0)
        output = np.concatenate((output, np.reshape(np.mean(avg, axis=0), (1, -1))), axis=0)
        output_sd = np.concatenate((output_sd, np.reshape(np.std(avg, axis=0), (1, -1))), axis=0)

    fmt = '%d %7f %7f %d %7f %7f %d'
    Path(base_path + output_folder).mkdir(parents=True, exist_ok=True)
    np.savetxt(base_path + output_folder+algo+'-'+digit+'-'+str(run_start).zfill(2)+'-'+str(run_end).zfill(2)+'-performance_avg.txt', output, fmt=fmt)

    fmt = '%d %7f %7f %d %7f %7f %7f'
    np.savetxt(base_path + output_folder+algo+'-'+digit+'-'+str(run_start).zfill(2)+'-'+str(run_end).zfill(2)+'-performance_sd.txt', output_sd, fmt=fmt)
    print("done")

save_max_rows()
sav_averages()