import librosa
import numpy as np
import json
import pyaudio
import wave
import serial
import time

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from playsound import playsound

#---- load data -----
PATH_JSON = "data.json"
f = open(PATH_JSON)
data = json.load(f)

#----- latihan ------
#pemisahan data
x = np.array(data['MFCCs'])[:,35,:]
y = np.array(data['labels'])

#split 80% train, 20% test
x_train = x[np.r_[:20,30:50]]
x_test = x[20:30]
y_train = y[np.r_[:20,30:50]]
y_test = y[20:30]

#knn
knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform')
knn.fit(x_train,y_train)
predictionKNN = knn.predict(x_test)
accuracyKNN = metrics.accuracy_score(y_test, predictionKNN)
print('accuracy KNN: ', accuracyKNN)

#-------- prediksi-------------
#fungsi rekam
def record_audio():
    filename = "record.wav"
    # set the chunk size of 2048 samples
    chunk = 2048
    # sample format
    FORMAT = pyaudio.paInt16
    # mono, ubah ke 2 untuk stereo
    channels = 1
    sample_rate = 22050
    record_seconds = 1

    #mulai record
    print("Record start in 3...")
    time.sleep(1)
    print("Record start in 2...")
    time.sleep(1)
    print("Record start in 1...")
    time.sleep(1)
    # initialize PyAudio object
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                channels=channels,
                rate=sample_rate,
                input=True,
                output=True,
                frames_per_buffer=chunk)
    frames = []
    print("Recording...")
    for i in range(int(sample_rate / chunk * record_seconds)):
        data = stream.read(chunk)
        # if you want to hear your voice while recording
        # stream.write(data)
        frames.append(data)
    print("Finished recording.")
    # stop and close stream
    stream.stop_stream()
    stream.close()
    # terminate pyaudio object
    p.terminate()

    # save audio file
    # open the file in 'write bytes' mode
    wf = wave.open(filename, "wb")
    # set the channels
    wf.setnchannels(channels)
    # set the sample format
    wf.setsampwidth(p.get_sample_size(FORMAT))
    # set the sample rate
    wf.setframerate(sample_rate)
    # write the frames as bytes
    wf.writeframes(b"".join(frames))
    # close the file
    wf.close()

#fungsi prediksi
def predict_audio():
    test_file = "record.wav"
    num_mfcc = 13
    n_fft=2048
    hop_length = 512
    SAMPLES_TO_CONSIDER = 22050 # 1 detik

    signal, sample_rate = librosa.load(test_file)
    if len(signal) >= SAMPLES_TO_CONSIDER:

        # memastikan konsistensi dari panjang sinyal
        signal = signal[:SAMPLES_TO_CONSIDER]

    # extraksi MFCCs
    MFCCs = librosa.feature.mfcc(signal, sample_rate,n_mfcc=num_mfcc, n_fft=n_fft,hop_length=hop_length)

    #prediksi
    test_value = np.array(MFCCs.T[35]).reshape(1,-1)
    test_predictionKNN = knn.predict(test_value)
    print('prediction KNN: ', test_predictionKNN)
    return test_predictionKNN
#--------- arduino communication ------------------
nodeMcu = serial.Serial('COM4', 9600)

def led_on_off(val):
    if val == 1:
        time.sleep(0.1)
        nodeMcu.write(b'H')
        time.sleep(0.1)
        print("LED is ON...")
    elif val == 0:
        time.sleep(0.1)
        nodeMcu.write(b'L')
        time.sleep(0.1)
        print("LED is OFF...")
    else:
        print("unknown led value.")

#---------------- start prediction --------------------
def start_predict():
    user_input = input("\n Type '1' to start predict or 'quit' : ")
    if user_input =="1":
        time.sleep(0.1)
        record_audio()
        led_on_off(predict_audio())
        print("Prediction used was KNN")
        start_predict()
    elif user_input =="quit" or user_input == "q":
        time.sleep(0.1)
        led_on_off(0)
        print("Program Exiting")
        nodeMcu.close()
    else:
        print("Invalid input. Type '1' to start predict or 'quit'.")
        start_predict()

time.sleep(2)
start_predict()
#
