import time

import mne
import numpy as np
from joblib import load
from keras.models import load_model

import calculation as cal

file = "test2.edf"
channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
random = 77
conv_model_filepath_arousal = "trained_models/CONV_FEATURES5_NO_DROPOUT_AROUSAL_2019_09_05_04_30_06"
conv_model_filepath_valence = "trained_models/CONV_FEATURES5_NO_DROPOUT_VALENCE2019_09_05_03_48_50"
knn_model_filepath_arousal = 'trained_models/knn_arousal.joblib'
knn_model_filepath_valence = 'trained_models/knn_valence.joblib'

windows = 29
frequency = 128
window_size = 4  # in seconds


def process(start, end):
    edf = mne.io.read_raw_edf(file, preload=True)
    edf.pick_channels(channels)
    montage = mne.channels.read_montage(kind='loc', path="./")
    edf.set_montage(montage)

    # EOG artifacts removal
    # filter for ICA algorithm because of its sensitivity to low-frequency drifts
    edf.filter(1, None, None, n_jobs=1, fir_design='firwin')
    ica = mne.preprocessing.ICA(random_state=random)
    ica.fit(edf, picks=mne.pick_types(edf.info, eeg=True, meg=False))
    # ica.plot_sources(inst=edf)
    ica.exclude = [1]  # fixme

    ica.apply(edf)
    # edf.plot()

    # bandpass filter (similar to DEAP data processing)
    edf.filter(4, 45, None, n_jobs=1, fir_design='firwin')

    # common average reference
    edf.set_eeg_reference('average')

    conv_data = list()
    knn_data = list()

    for channel in channels:
        original_data = np.array(edf.get_data(channel, start, end)).flatten()

        # Normalisation
        all_data = np.array(edf.get_data(channel, 0, end)).flatten()

        # min = original_data.min()
        # divider = original_data.max() - min
        min = all_data.min()
        divider = all_data.max() - min
        data_normalised = (original_data - min) / divider

        # AMR for KNN
        amr_data = cal.calculate_dwt(cal.calculate_amr(data_normalised))

        # Convolution - order of channels should be kept

        data = cal.calculate_dwt(data_normalised)
        energy = cal.calculate_energy(data)
        power = cal.calculate_power(energy, len(data))
        entropy = cal.calculate_entropy(data)
        mean = cal.calculate_mean(data)
        st_dev = cal.calculate_standard_deviation(mean, data)

        conv_data.append(energy)
        conv_data.append(power)
        conv_data.append(entropy)
        conv_data.append(mean)
        conv_data.append(st_dev)

        knn_data.append(cal.calculate_energy(amr_data))
        knn_data.append(cal.calculate_entropy(amr_data))

    conv_input = np.array(conv_data).reshape(70, 1)
    knn_input = np.array(knn_data)

    return conv_input, knn_input


def main():
    experiment_start = time.time()
    end_time = time.time()

    conv_model_arousal = load_model(conv_model_filepath_arousal)
    conv_model_valence = load_model(conv_model_filepath_valence)

    knn_arousal = load(knn_model_filepath_arousal)
    knn_valence = load(knn_model_filepath_valence)

    step = 2 * frequency
    start = 0
    end = window_size * frequency
    # windows = 29

    print("Starting experiment")
    input("Press Enter to continue...")

    filename = 'result_{}.txt'.format(time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime()))
    file = open(filename, 'w')
    file.write('conv_arousal,conv_valence,knn_arousal, knn_valence\n')
    time.sleep(4.01)

    time_print = ""

    for i in range(windows):
        start_time = time.time()

        conv_input, knn_input = process(start, end)

        conv_pred_arousal = (conv_model_arousal.predict(np.array([conv_input]))[0]).argmax()
        conv_pred_valence = (conv_model_valence.predict(np.array([conv_input]))[0]).argmax()

        knn_pred_arousal = knn_arousal.predict(np.array([knn_input]))[0]
        knn_pred_valence = knn_valence.predict(np.array([knn_input]))[0]

        file.write("{},{},{},{}\n".format(conv_pred_arousal, conv_pred_valence, knn_pred_arousal, knn_pred_valence))

        start += step
        end += step

        end_time = time.time()
        gap = 2. - (end_time - start_time)
        sleep = gap if gap > 0 else 0
        time_print += "{} ".format(end_time - start_time)
        # time.sleep(sleep)

    file.close()
    print(time_print)
    print("Experiment took: {}".format(end_time - experiment_start))


if __name__ == "__main__":
    main()
