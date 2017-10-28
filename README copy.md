# Classification of Cat and Dog .wav files. 

In this project, I obtained a corpus of cat vocalizations and dog vocalizations from https://www.kaggle.com/mmoreaux/audio-cats-and-dogs   

I will use the librosa and sklearn libraries heavily for the audio and modeling of this project. 

## Data 
All the WAV files contains 16KHz audio and have variable length.

I took out  some of the files that had both cat and dog sounds present or music in the background which left us with 147 cat sounds and 108 dog sounds. 

# Visualization

For some initial visualization, load 
## spectrograms_catdog 

This notebook allows you to plot the raw wavform (using amplitude (Root Mean Squared)) vs. time, along with the spectrogram. The function calculates the Short Time Fourier Transform (STFT) with a window size of 1024 and plots the real values (magnitude) on a log-frequency scale vs. time (same units as the raw wave for comparison). 

At the bottom there is a little code if you want to plot only a portion of the file to 'zoom in' on long files. 

There are some interesting differences and similarities that I pointed out in the markdown above each example. (i.e. harmonic structure tends to be the most telling difference rather than pure frequency alone). 

# Exploratory Data Analysis

## cat_dog_SQL_readWav

I stored the location (path) of the files, and file ID (number associated with each sound) in a SQL table on my EC2 instance (Amazon Web Services). Then, I pulled from this table to create the list of paths and list of IDs to read in for processing. 

Once those dataframes are loaded, they can be passed to a function that will take the 1-D FFT (Fast Fourier Transform) of each file and store the FFT values (magnitude) and the corresponding frequency in a binary file along with the numpy array of the raw wave. 

Then,  in order to put the files into one matrix without any Nans, we have 2 choices: 
 
 1. cut off all the files at some lenghth (i.e. 2 seconds). This is problematic because you might not even get the animal sound if you arbitrarily cut them from beginning or end etc. So, I didn't do this, as I didn't want to have to find where the sound was or god-forbid go in and cut files by hand. 
 
 2. Take the frequencies from our 'shortest' FFT (the file that had the smallest number of frequencies present).  I knew that this would not perfectly correlate with the size (in KB) of the file, but figured that it would be close. Warning: if you aren't familiar with Fourier Transforms this likely won't make sense and that's totally fine. You can still use it because there's nothing to tweak. 
     
    
    step1: I took a small sample of 10 of the smallest wav files from the corpus (does not matter if cat or dog), and did the FFT, and output the dataframe conatining all the freq. arrays.  Just tweak the fuxn to return the df you want. 
     
     step2: then looked at the lengh of the freq vectors and saved the shortest one as my "master_frequencies" 

    These master frequencies are now going to be our template for how we 'resample' the rest of the files. 
 
     step 3: in the resample_freq function, we load our master frequencies along with a cat or dog FFT and freq file. 
     
     step 4: Then, we find the frequencies that are the closest to the frequencies in the master freqs and get the index.   
     
     step 5: Take this index to pull the corresponding magnitude values (fft values) and we now have something that is comparable!   
     
     step 6: save as n_fft which is the newfft (sampled).   This technique allows us to have a large dataframe where each row is a cat/dog sample and each column represents roughly the same requency, each value is that sample's power (magnitude) at that frequency. 

The last thing to do is take the cat and dog data frames, transform them so that we have samples= rows and frequency (these now become our features) = columns and add a 'target' column that tell us which category the row belongs to (cat = 0, dog = 1). 

These data frames get saved as pickle files in the location you put into the 'path' variable at the top of the notebook. 

# Dimensionality Reduction

### train_cat_dog

As you may have noticed, we have over 10,000 features!  The first thing we do after opening the file, is plot some histograms so that you can get an idea of what we are dealing with. 

That's a lot, so we will use Principal Components Analysis (PCA) to reduce the dimensions to something less computationally expensive. 

First we need to do a stratified (because classes are unbalanced) shuffle (because all the cats are together and dogs are together in the dataframe) split into a Training set and a Test set. 

Then, we use standard scaler to scale the data (this is critical for PCA). 

Plot the test-train split of a few features for a sanity check to make sure it looks like things have been evenly samples from the 2 classes. 

Then do the PCA with a large number of components. 
Why? because we need to see how much each component is contributing to the overall variance. So, plot the ratio of explained variance for each component along with the cumsum.  Note that if we take the first 75 components we will account for approx 87% of the variance and we COULD go with this number of components. However, this is probably not the best approach because after component 5 or 6 the contribution that each component makes is miniscule and will likely just add noise if we include them in our model.  So, I chose to take the first 6 components which explained approximately 70% of the variance in our features. 

Run the PCA again using n_components=6 on the Training set. Fit, transform, plot. 

Notice in this plot that we get some pretty good separation for one of the classes (dogs). This is what we are after. 

Then run the model on the test set, plot. 
It's hard to see the separation here since the test set is pretty small, but it looks very similar to the training which is good. 

I tried Kernel PCA to see if it gave me a better separation and as you an see, it just pushed everything together, so this was abandoned.

now we can take these reduced training and test sets and try some different classification models. 

# Classification models 

All of these models come out of the sklearn library which is beautiful. 
First we run a dummy classifier that predicts 'cat' every single time and use this as our base line.  Essentially, we want to see if our models can perform better than this. 

Now, we try a suite of models, output their scores and compare them using a roc curve. 
KNN with varying Ks to see which one gives us the highest accuracy (it's 5 for us) 
Logistic Regression
Gaussian Naive Bayes
SVC with Poly kernel (remember our cluster was not really separable with a straight line)
Decision Tree Classifier
RandomForest Classifier (2000 Trees) 

From the ROC curve we see that Gaussian Naive Bayes and Logistic Regression were the best. 

There are more things we could do to improve the score, too. Future directions would be to tweak the PCA, use Cross Validation, try PCA on each class, then put them back together for the modeling. Or, add more data (of course).  The small data set was a limitation but to my surpise, the model still worked pretty well.  


