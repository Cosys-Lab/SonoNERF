# SonoNERF
 
We introduce SonoNERFs, a novel approach that adapts Neural Radiance Fields (NeRFs) to model and understand the echolocation process in bats, focusing on the challenges posed by acoustic data interpretation without phase information. Leveraging insights from the field of optical NeRFs, our model, termed SonoNERF, represents the acoustic environment through Neural Reflectivity Fields. This model allows us to reconstruct three-dimensional scenes from echolocation data, obtained by simulating how bats perceive their surroundings through sound. By integrating concepts from biological echolocation and modern computational models, we demonstrate the SonoNERF’s ability to predict echo spectrograms for unseen echolocation poses and effectively reconstruct a mesh-based and energy-based representation of complex scenes. Our work bridges a gap in understanding biological echolocation and proposes a methodological framework that provides a first order model on how scene understanding might arise in echolocating animals. We demonstrate the efficacy of the SonoNERF model on three scenes of increasing complexity, including some biologically relevant prey-predator interactions.

This repository contains the Matlab source code to create the SonoNERF model and evaluate it based on simulation data. 

Paper: https://www.biorxiv.org/content/10.1101/2024.04.20.590416v1

## Publication
We kindly ask to cite our paper if you find this codebase usefull:
```
@article {SonoNERF2024,
	author = {Jansen, Wouter and Steckel, Jan},
	title = {SonoNERFs: Neural Radiance Fields applied to Biological Echolocation Systems allow 3D Scene Reconstruction Through Perceptual Prediction},
	elocation-id = {2024.04.20.590416},
	year = {2024},
	doi = {10.1101/2024.04.20.590416},
	URL = {https://www.biorxiv.org/content/early/2024/04/25/2024.04.20.590416},
	eprint = {https://www.biorxiv.org/content/early/2024/04/25/2024.04.20.590416.full.pdf},
	journal = {bioRxiv}
}
```

## Usage

### Dependencies
 - Matlab 2024a or higher
   - [SonoTraceLab](https://github.com/Cosys-Lab/SonoTraceLab) and its dependencies
   - Parallel Computing Toolbox
   - Signal Processing Toolbox
   - Deep Learning Toolbox
   - Image Processing Toolbox
   - Phased Array System Toolbox
   - Lidar Toolbox

### Code
The underlying source code is provided as well as five scripts to run each step of the process as shown in the paper. 
 - r1_generateDatasetSonoNERF.m: Generate the simulation data using [SonoTraceLab](https://github.com/Cosys-Lab/SonoTraceLab) based on a scenario 3D model.
 - r2_convertDataSetNERF_STFT.m: Convert the simulation data by calculating the STFT of all the measurements and saving it in a datastore format ready for training.
 - r3_trainSonoNERF.m: Using the datastore and the bat HRTF, train the NERF network and create the model. 
 - r4_evaluateSonoNERF.m: Evaluate the trained model and run inference a display the predicted spectrograms and the maximum intensity projection of the reconstruction.
 - r5_makeSonoNerfResultPlot.m: Evaluate the trained model and run inference and display the 3d scene of the scenario and the predicted measurement locations, the predicted spectograms and the original ground truth. and the reconstructed isosurface of the reconstruction.

### Data
As the bat HRTF and 3D head model used in the paper are proprietary, a generic bat HRTF and 3d model is provided within the dataset. Three scenarios used in the paper are provided: the UA logo, the dragonfly on the leaf, and the three spheres. These STL 3D models can be found in the dataset. For each of them the simulated data (generated with the first script) comes already pre-generated. 

## License
This library is provided as is, will not be actively updated and comes without warranty or support.
Please contact a Cosys-Lab researcher to get more in depth information or if you wish to collaborate.
SonoNERF is open source under the MIT license, see the [LICENSE](LICENSE) file.

## Open-Source libraries included in this project
 - Recursive Zonal Equal Area (EQ) Sphere Partitioning Toolbox by Paul Leopardi for the University of New South Wales [(link)](https://github.com/penguian/eq_sphere_partitions)
 - Progress bar by HyunGwang Cho [(link)](https://www.mathworks.com/matlabcentral/fileexchange/121363-progress-bar-cli-gui-parfor?s_tid=srchtitle)