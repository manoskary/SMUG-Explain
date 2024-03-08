# SMUG-Explain
SMUG-Explain stands for Symbolic MUsic Graph Explanations. 
It is a framework for generating and visualizing explanations of graph neural networks 
applied to arbitrary prediction tasks on musical scores. SMUG-Explain allows the user to 
visualize the contribution of input notes (and note features) to the network output, 
directly in the context of the musical score. We provide an interactive interface based 
on the music notation engraving library Verovio. We showcase the usage of SMUG-Explain on 
the task of cadence detection in classical music. 


## Installation and Usage

To use SMUG-Explain you don't need to install anything. 
You need to open the `index.html` file in your browser (for example by double-clicking) and follow the instructions.
You will need to provide a MEI file and a JSON file with the explanations or use one of the already available in the static folder.


## Generating Explanations for Cadence Detection

To generate explanations for the `cadence` detection model, you need to follow the example notebook in the `notebooks` folder.
We recommend running the notebook as a Google Colab notebook, as it will handle the installation of the required dependencies for you.
To run the model several requirements are needed, therefore the process is a bit more complex than the one for the web interface.

The example demonstrates how to use a graph-based `cadence` detection model to explain the predictions made by the model. The `cadence` model is a deep learning model that detects the cadence of a musical score. The model is trained on a dataset of musical scores and their corresponding cadences. 
The model takes a musical score as input and outputs a cadence label for every note, export the score to MEI and generate a JSON file with the explanations.


### Some Remarks

The input score can be of any of the following formats:
- MEI
- MusicXML
- MIDI
- MuseScore
- Kern

But it needs to contain a single part to export a readable representation on the veroio interface.

## Aknowledgments

The Web interface is based on the [Music Graph Visualizer](https://github.com/fosfrancesco/musgviz) by Francesco Foscarin.
The Cadence Detection model is based on the [Cadence Detection](https://github.com/manoskary/cadet) model by Emmanouil Karystinaios.




