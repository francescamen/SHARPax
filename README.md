# SHARPax

Algorithms for human activity recognition with a commercial IEEE 802.11ax router @ 5 GHz, 80 MHz of bandwidth.

This repository contains the reference code for the article [''Toward Integrated Sensing and Communications in IEEE 802.11bf Wi-Fi Networks''](https://arxiv.org/abs/2212.13930). The algorithms are an adaptation to the case of 802.11ax devices of the original [''SHARP algorithm''](https://ieeexplore.ieee.org/document/9804861).

If you find the project useful and you use this code, please cite our articles:
```
 @article{meneghello2023toward,
  author = {Meneghello, Francesca and Chen, Cheng and Cordeiro, Carlos and Restuccia, Francesco},  
  journal={IEEE Communications Magazine}, 
  title = {{Toward Integrated Sensing and Communications in IEEE 802.11bf Wi-Fi Networks}},
  year = {2023},
  volume={},
  number={},
  pages={}
  }
```

```
@article{meneghello2022sharp,
  author={Meneghello, Francesca and Garlisi, Domenico and Dal Fabbro, Nicol\o' and Tinnirello, Ilenia and Rossi, Michele},
  journal={IEEE Transactions on Mobile Computing}, 
  title={{SHARP: Environment and Person Independent Activity Recognition with Commodity IEEE 802.11 Access Points}}, 
  year={2022},
  volume={},
  number={},
  pages={1-16}
  }
```

## How to use
Clone the repository and enter the folder with the python code:
```bash
cd <your_path>
git clone https://github.com/francescamen/SHARPax
```

Download the input data from [here](https://drive.google.com/file/d/1JbWNV3fMAF-26SJfqX0EohkkrUe2SeC4/view?usp=sharing) and unzip the file. 
For your convenience, you can use the ```input_files/processed_files/``` folder inside this project to place the files but the scripts work whatever is the source folder.

The dataset contains Wi-Fi channel frequency response (CFR) data collected in an IEEE 802.11ax network through [AX CSI](https://ans.unibs.it/projects/ax-csi/). 
The network consists of two ASUS RT-AX86U Wi-Fi routers operating on the IEEE 802.11ax channel number 157 using the OFDMA resource unit RU1-996, i.e., with a bandwidth of 80 MHz and 996 data sub-channels. The CFR is obtained for each packet collected by the receiver device while a person acts as an obstacle for the transmission by performing different activities. 
The considered movements are the following: walking (W) or running (R) around, and staying (S) in place.
The CFR data for the empty room (E) is also provided.
The complete description of the dataset can be found in the reference paper and in the IEEE DataPort repository.

The code for SHARPax is implemented in Python and can be found in the ```Python_code``` folder inside this repository. The scripts to perform the processing are described in the following, together with the specific parameters.

### Phase sanitization
The following three scripts encode the phase sanitization algorithm detailed in Section 3.1 of [meneghello2022sharp](https://ieeexplore.ieee.org/document/9804861)].
```bash
python CSI_phase_sanitization_signal_preprocessing.py <'directory of the input data'> <'process all the files in subdirectories (1) or not (0)'> <'name of the file to process (only if 0 in the previous field)'> <'number of spatial streams'> <'number of cores'> <'number of OFDMA sub-channels including control sub-channels'> <'index where to start the processing for each stream'> 
```
e.g., python CSI_phase_sanitization_signal_preprocessing.py ../input_files/processed_files/ 1 - 1 4 1024 0

```bash
python CSI_phase_sanitization_H_estimation.py <'directory of the input data'> <'process all the files in subdirectories (1) or not (0)'> <'name of the file to process (only if 0 in the previous field)'> <'number of spatial streams'> <'number of cores'> <'index where to start the processing for each stream'> <'index where to stop the processing for each stream'> 
```
e.g., python CSI_phase_sanitization_H_estimation.py ../input_files/processed_files/ 0 R2_P1 1 4 0 -1

```bash
python CSI_phase_sanitization_signal_reconstruction.py <'directory of the processed data'> <'directory to save the reconstructed data'> <'number of spatial streams'> <'number of cores'> <'number of OFDMA sub-channels including control sub-channels'> <'index where to start the processing for each stream'> <'index where to stop the processing for each stream'> 
```
e.g., python CSI_phase_sanitization_signal_reconstruction.py ./phase_processing/ ./processed_phase/ 1 4 1024 0 -1

### Doppler computation
The following script computes the Doppler spectrum as described in Section 3.2 of [meneghello2022sharp](https://ieeexplore.ieee.org/document/9804861)].

```bash
python CSI_doppler_computation.py <'directory of the reconstructed data'> <'sub-directories of data'> <'directory to save the Doppler data'> <'starting index to process data'> <'end index to process data (samples from the end)'> <'number of packets in a sample'> <'number of packets for sliding operations'> <'noise level'> <--bandwidth 'bandwidth'> <--sub_band 'sub band to consider (in {1, 2} for 40 MHz, in {1, 2, 3, 4} for 20 MHz)'> <-- sub_sampling 'sub sampling factor in {1, ..., 6}'>
```
e.g., python CSI_doppler_computation.py ./processed_phase/ E1,E2,E3,E4,R1_P1,R2_P1,R3_P1,R4_P1,S1_P1,S2_P1,S3_P1,S4_P1,W1_P1,W2_P1,W3_P1,W4_P1 ./doppler_traces/ 200 200 25 1 -1.5 --bandwidth 40 --sub_band 2 --sub_sampling 1

Helper function to visualize the Doppler traces:
```bash
python CSI_doppler_plot_antennas.py <'directory of the Doppler data'> <'sub-directories of data'> <'number of packets in a sample'> <'number of packets for sliding operations'> <'end index to visualize data (samples from the end)'> <'noise level'> <--bandwidth 'bandwidth'> <--sub_band 'sub band to consider (in {1, 2} for 40 MHz, in {1, 2, 3, 4} for 20 MHz)'> <-- sub_sampling 'sub sampling factor in {1, ..., 6}'>
```
e.g., python CSI_doppler_plot_antennas.py ./doppler_traces/ E1,E2,E3,E4,R1_P1,R2_P1,R3_P1,R4_P1,S1_P1,S2_P1,S3_P1,S4_P1,W1_P1,W2_P1,W3_P1,W4_P1 31 1 -1 -1.5

### Dataset creation
- Create the datasets for cross validation
```bash
python CSI_doppler_create_datasets_cross_val.py <'directory of the Doppler data'> <'sub-directories, comma-separated'> <'number of packets in a sample'> <'number of packets for sliding operations'> <'number of samples per window'> <'number of samples for window sliding'> <'labels of the activities to be considered'> <'number of streams * number of antennas'>
```
  e.g., python CSI_doppler_create_datasets_cross_val.py ./doppler_traces/ 4 2 1 256 24 E,S,W,R 4 -1.5 --bandwidth 40 --sub_band 2 --sub_sampling 1

### Train the learning algorithm for HAR and assess the performance (4-fold cross-validation)
```bash
python CSI_network.py <'directory of the datasets'> <'sub-directories for training, comma-separated'> <'sub-directories for validation, comma-separated'> <'sub-directories for test, comma-separated'> <'length along the feature dimension (height)'> <'length along the time dimension (width)'> <'number of channels'> <'number of samples in a batch'> <'number of streams * number of antennas'> <'name prefix for the files'> <'activities to be considered, comma-separated'> <--bandwidth 'bandwidth'> <--sub-band 'index of the sub-band to consider (for 20 MHz and 40 MHz)'> 
```
e.g., 
python CSI_network.py ./doppler_traces/dataset_train_val_test/ 1,2 3 4 100 256 1 32 4 network E,S,W,R --bandwidth 80 --sub_band 1 --sub_sampling 1

- Compute and visualize the performance metrics using the output files
```bash
python CSI_network_metrics.py <'sub-directories for training, comma-separated'> <'sub-directories for validation, comma-separated'> <'sub-directories for test, comma-separated'> <'activities to be considered, comma-separated'> <'name prefix for the files'> <--bandwidth 'bandwidth'> <--sub-band 'index of the sub-band to consider (for 20 MHz and 40 MHz)'> 
```
  e.g., python CSI_network_metrics.py 1,2 3 4 E,S,W,R 091122 --bandwidth 80 --sub_band 1 --sub_sampling 3
 
- Compute and save the performance metrics using the output files for plotting
```bash
python CSI_network_metrics_cross_val.py <'activities to be considered, comma-separated'> <'number of streams * number of antennas'> <'names prefix of the files, comma-separated'> <'number of directories considered'> <'number of directories for training'> <'number of directories for validation'> <--bandwidth 'bandwidth'> <--sub-band 'index of the sub-band to consider (for 20 MHz and 40 MHz)'> 
```
  e.g., python CSI_network_metrics_cross_val.py E,S,W,R 4 trial1,trial2,trial3,trial4,trial5,trial6,trial7,trial8,trial9 4 2 1 --bandwidth 80 --sub_band 1 --sub_sampling 1

- Plot the performance metrics
```bash
python CSI_network_metrics_cross_val_plots_different_bandwidth.py <'activities to be considered, comma-separated'> <'names prefix of the files, comma-separated'>
```
  e.g., python CSI_network_metrics_cross_val_plots_different_bandwidth.py E,S,W,R trial1,trial2,trial3,trial4,trial5,trial6,trial7,trial8,trial9

```bash
python CSI_network_metrics_cross_val_plots_different_samplings.py <'activities to be considered, comma-separated'> <'names prefix of the files, comma-separated'>
```
  e.g., python CSI_network_metrics_cross_val_plots_different_samplings.py E,S,W,R trial1,trial2,trial3,trial4,trial5,trial6,trial7,trial8,trial9

```bash
python CSI_network_metrics_cross_val_plots_different_samplings_combined.py <'activities to be considered, comma-separated'> <'names prefix of the files, comma-separated'>
```
  e.g., python CSI_network_metrics_cross_val_plots_different_samplings_combined.py E,S,W,R trial1,trial2,trial3,trial4,trial5,trial6,trial7,trial8,trial9

### Parameters
The results of the article are obtained with the parameters reported in the examples. For convenience, the repository also contains four pre-trained networks, i.e.,  
``091122_train_[2 3]_val_[1]_test_[4]_E,S,W,R__bandw20_RU4_sampling1_network.h5``,  
``091122_train_[2 3]_val_[1]_test_[4]_E,S,W,R__bandw40_RU2_sampling1_network.h5``,  
``091122_train_[2 3]_val_[1]_test_[4]_E,S,W,R__bandw80_RU1_sampling1_network.h5``,  
``091122_train_[2 3]_val_[1]_test_[4]_E,S,W,R__bandw80_RU1_sampling3_network.h5``.

### Python and relevant libraries version
Python >= 3.8.5  
TensorFlow >= 2.7.0  
Numpy >= 1.21.5  
Scipy = 1.4.1  
Scikit-learn = 0.23.2  
OSQP >= 0.6.1

## Contact
Francesca Meneghello
francesca.meneghello.1@unipd.it
github.com/francescamen