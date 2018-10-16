# GANGogh

Blog post: https://towardsdatascience.com/gangogh-creating-art-with-gans-8d087d8f74a1

Note: Code heavily inspired and built off of the improved wasserstein GAN training code available and found at: https://github.com/igul222/improved_wgan_training

## Usage:

### Step 1 - Gather training data
We used training data from wikiart.org, but any training data will do. It's prefered to download this training data from [this torrent](http://academictorrents.com/details/1d154cde2fab9ec8039becd03d9bb877614d351b) or the [Google Drive file](https://drive.google.com/file/d/1yHqS2zXgCiI9LO4gN-X5W18QYXC5bbQS/view?usp=sharing). If both of those fail, consider using scape_wiki.py as a last resort.
<br><br>
### Step 2 - Prepare the training data
Use picStuff.py to create image data set of 64x64 pieces of art scraped from wikiart. Take note of the `root` and `PATH` variables and modify accordingly.
<br><br>
### Step 3 - Modify files
Update the path to the dataset in wikiartGenre.py. Also, update the `styles` variable dictating the number of training images per genre. If using the traning data set linked, above, use the following:
```python
styles = {'abstract': 14999,
          'animal-painting': 1798,
          'cityscape': 6598,
          'figurative': 4500,
          'flower-painting': 1800,
          'genre-painting': 14997,
          'landscape': 15000,
          'marina': 1800,
          'mythological-painting': 2099,
          'nude-painting-nu': 3000,
          'portrait': 14999,
          'religious-painting': 8400,
          'still-life': 2996,
          'symbolic-painting': 2999}
```
<br><br>
### Step 3 - Make art!
Run GANGogh.py
