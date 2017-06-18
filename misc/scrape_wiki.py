"""
This file contains a script that scrapes the wikiart database for a certain specified amount of photos for given genres. The amount of images downloaded can be controlled in the genres dictionary. This method is up-to-date with respect to url names and image amounts as of June 2017.
"""

import os
import bs4
import urllib
import urllib.request
from bs4 import BeautifulSoup
import itertools
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing

# A list of genres hosted on wikiart.org as well as the number of pages to pull images from, numbers were set from manual inspection and are only approximations of how many pages each genre contains
genres = [('portrait',250),
        ('landscape',250),
        ('genre-painting',250),
        ('abstract',250),
        ('religious-painting',140),
        ('cityscape',110),
        ('figurative',75),
        ('still-life',50),
        ('symbolic-painting',50),
        ('nude-painting-nu',50),
        ('mythological-painting',35),
        ('marina',30),
        ('flower-painting',30),
        ('animal-painting',30)]




#Access the html of the page given a genre and pagenumber that are used to generate a url, from this html find the urls of all images hosted on the page using page layout as of June 2017, return a list of alls urls to paintings
def soupit(j,genre):
    try:
        url = "https://www.wikiart.org/en/paintings-by-genre/"+ genre+ "/" + str(j)
        html = urllib.request.urlopen(url)
        soup =  BeautifulSoup(html)
        found = False
        urls = []
        for i in str(soup.findAll()).split():
            if i == 'data':
                found = True
            if found == True:
                if '}];' in i:
                    break;
                if 'https' in i:
                    web = "http" + i[6:-2]
                    urls.append(web)
                    j = j+1
        return urls
    except Exception as e:
        print('Failed to find the following genre page combo: '+genre+str(j))


#Given a url for an image, we download and save the image while also recovering information about the painting in the saved name depending on the length of the file.split('/') information (which corresponds to how much information is available)

def dwnld(web,genre):
    i,file = web
    name = file.split('/')
    savename = ''
    if len(name) == 6:
        savename = genre+"/"+ name[4] + "+" + name[5].split('.')[0] +".jpg"
    if len(name) == 5:
        savename = genre+"/"+name[4].split('.')[0]+".jpg"
    if len(name) == 7:
        savename = genre+"/"+ name[5] + "+" + name[6].split('.')[0] +".jpg"
        
    print(genre + str(i))
    #If we get an exception in this operation it is probably because there was a nonstandard unicode character in the name of the painting, do some fancy magic to fix this in the exception handling code
    try:
        urllib.request.urlretrieve(file,savename)
    except Exception:
        ofile = file
        file = urllib.parse.urlsplit(file)
        file = list(file)
        file[2] = urllib.parse.quote(file[2])
        file = urllib.parse.urlunsplit(file)
        try:
            urllib.request.urlretrieve(file,savename)
            print('Suceeded on second try for '+ file)
        except Exception:
            print('We failed on the second try for ' + file)


#We can run both the url retrieving code and the image downloading code in parallel, and we set up the logic for that here
def for_genre(genre,num):
    pool = ThreadPool(multiprocessing.cpu_count()-1)
    nums = list(range(1,num))
    results = pool.starmap(soupit,zip(nums,itertools.repeat(genre)))
    pool.close()
    pool.join()
    
    #build up the list of urls with the results of all the sub-processes that succeeded in a single list
    new_results = []
    for j in results:
        if j:
            for i in j:
                new_results.append(i)
    
    pool = ThreadPool(multiprocessing.cpu_count()-1)
    pool.starmap(dwnld,zip(enumerate(new_results),itertools.repeat(genre)))
    pool.close
    pool.close()
        
if __name__ == '__main__':
    for (a,b) in genres:
        if not os.path.exists("./"+a):
            os.mkdir(a)
        for_genre(a,b)

    
                

