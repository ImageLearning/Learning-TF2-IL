

# Lesson Plan

We will be building both Python / Tensorflow / ML code, and writing "supporting code" to show good workflow / GitHub practices. 

Lined up  : https://github.com/ImageLearning/Learning-TF2-IL/tree/master/Lessons with below, ended up "off by 1"

0. Setup Python & Tensorflow on Mac, Windows, and Windows-GPU : https://youtu.be/vZtfbdmXN_s
> In this lesson we will get setup to write code in Python and TF2, we will also run some existing code and do Q&A
1. Basic Python Programming & GitHub
> In this lesson, we will learn some python and check in our code to github as we go
- Create a new GitHub Repo to work out of
- Check it out locally 
- Add a .gitignore file
- Add a ReadMe.md file
- basic Markdown syntax
- Git Commit
- Git Sync
- quick file layout
- Folder layout
    - Docs folder
    - Source folder
- GitHub Pages
- Branch 
- Add a file that is too large (our model) and Git Reset to undo our last commit
- Git Add . VS git add * https://gist.github.com/dsernst/ee240ae8cac2c98e7d5d
2. Basic ML & Actions & Your first Image recognizer
> In this lesson we will write our first Tensorflow / ML code. We will setup actions to confirm that the code we check in works and adhere to good principles (python linter). We'll end with a classic Image recognition sample
3. Data Prep Setup & GitHub Pages 
> In this lesson, we'll learn to identify if something is an octocat. Then we'll setup our own image recognition dataset with a variety of octocat stickers, and learn GitHub pages to publish our data set to. We'll also adjust our action to package our dataset for easy down by others.
4. Detecting Octocats with Tensorflow 2
> In this lesson, we'll take the data we prepared in lesson 4 and train a model to identify a few different octocats. Depending on if the community is involved and contributing, we may identify a lot. If there is time, and we get fancy, we could set it up as a local actions runner.
5. Polish on our Machine learning project & Pages & Actions
> In this lesson, we'll run Machine Learning a second time on our data set in a different way, to help differentiate between very similar octocats. We will also polish our Pages and Actions projects to make them easier for others to use, and see what is going on. Also advanced troubleshooting steps
6.  Port app to iOS + Actions
> We will port the recognizer to iOS, unclear if this is going to be moving the code, or just the trained model and CoreML. Also need to make sure the model we use in stage 4 or 5 works on iOS
7. Port app to Android + Actions
> We will port the recognizer to Android, unclear if this is going to be moving the code, or just the trained model. Also need to make sure the model we use in stage 4 or 5 works on iOS

**Output : ML**
- 

**Output : Supporting tools**
- Action : Resize Images
- Action : Package images for easy download 
- GitHub Pages hosting of Images and Downloads
    - Good First Issue : Display bounding rectangles 

## Lesson Format

Each Lesson starts with a sample folks can download to follow along
Each Lesson ends with 
- A completed sample folks can use in an Open Source repo
- A call to action to extend the Open Source ML sample
- A call to action to extend or enhance the "supporting tools"


### Tooling / Process

#### Windows Machine Prep
(streaming prep / machine learning exploration)

Since a portion of what I'm going to be doing involves a clean machine install of tools, that involve GPU configurations, and re-installing windows over and over again is "lots of fun" I'm using
- WinToUSB Professional : https://www.easyuefi.com/wintousb/comparison.html 
- one of these I have laying around, but if it turns out to be too small, i'll upgrade https://www.amazon.com/dp/B079NWJTGG/ref=twister_B079P763J5?_encoding=UTF8&psc=1 

this lets me clone my current "bare minimum" windows install to a bootable USB drive, boot from that and work from that at fast speeds (fast as the usb drive will let + data on my SSD raid)
and then blow it away and start over for the next practice run

I clone to the usb drive in .vhdx mode.
- before i start a run i save the .vhdx off to the side 
- after the run if i got a good run, i save the before and after .vhdx's along side the video 

end result may be a few hundred gigs of windows images, but I can always go back to the environment to re-record if i need to (until I need space)
