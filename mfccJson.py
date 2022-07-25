import librosa
import os
import json

dataset_path = r"G:/Dafi/dokumen/kuliah UB/semester 5/Pemrosesan Suara/Dataset Suara/ain"
json_path = "data.json"
num_mfcc = 13
n_fft=2048
hop_length = 512
SAMPLES_TO_CONSIDER = 22050 # 1 detik

def preprocess_dataset(dataset_path, json_path, num_mfcc=13, n_fft=2048,hop_length=512):

    data = {
    "mapping": [],
    "labels": [],
    "MFCCs": [],
    "files": []
    }

    # Menyusuri semua sub folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # memastikan penyusuran data berada di subfolder
        if dirpath is not dataset_path:

            # simpan label di mapping berdasarkan nama subfolder
            label = dirpath.split("/")[-1]
            data["mapping"].append(label)
            print("\nProcessing: '{}'".format(label))

            # proses semua file audio di subfolder, lalu simpan mfcc
        for f in filenames:
            file_path = os.path.join(dirpath, f)

            # load file audio dan memotongnya agar panjang sinyal dari file yg berbeda tetap konsisten
            signal, sample_rate = librosa.load(file_path)

            # memasukkan file audio dengan jumlah sampel yang lebih sedikit dari sebelumnya
            if len(signal) >= SAMPLES_TO_CONSIDER:

                # memastikan konsistensi dari panjang sinyal
                signal = signal[:SAMPLES_TO_CONSIDER]

            # extraksi MFCCs
            MFCCs = librosa.feature.mfcc(signal, sample_rate,n_mfcc=num_mfcc, n_fft=n_fft,hop_length=hop_length)

            # simpan data untuk mengetahui jalur penyusuran
            data["MFCCs"].append(MFCCs.T.tolist())
            data["labels"].append(i-1)
            data["files"].append(file_path)
            print("{}: {}".format(file_path, i-1))

        # simpan data dalam bentuk json
        with open(json_path, "w") as fp:
            json.dump(data, fp, indent=4)

if __name__ == "__main__":
    preprocess_dataset(dataset_path, json_path)
