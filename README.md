# NN Solubility predictor

## Overview
This project was my first predicting Neural Network. The idea was to train a neural network to learn the patterns in molecular SMILES strings and to teach it the patterns in solubility of the molecules. SMILES is a molecular string representation that is capable of keeping some notion of structure and atomic make-up of a given molecule. The dataset consisted of labelled molecules with their respective solubilities at room temperature.

## Challenges
This was the first project wherein the majority of challenges were centered around the expiramentation phase. Upon completing data preprocessing and model architecture, expiraments consistently shoed poor predictive performance. Dr. Esben Bjerrum recommended
1. Using different activation functions (ReLU for hidden layers and linear for output layer)
2. scaling the target (Y) values appropriately. 

## Sample Images
![](https://www.flinnsci.com/globalassets/flinn-scientific/all-product-images-rgb-jpegs/ap6901.jpg?v=05db0fa7349249ca9f9e6839bda1c6c3)

## Applications
Project Dissolved has potential uses in chemistry and pharmaceuticals:
* Predicting the solubility of different molecules with different states of matter
* Predicting the solubility of different molecules at different temperatures
* Predicting the solubility of bio-compatable molecules, such as drugs, implants, among other items

## Future
Project Dissolved is the first project I classified as complete, but failed. Future improvements include using pandas, matplotlib, and seaborn to better demonstrate and examine where the model fell short, as well as modifying the model architecture entirely to better suit the task of predicting solubilies of molecules.

Upwards and onwards, always and only! :rocket:
