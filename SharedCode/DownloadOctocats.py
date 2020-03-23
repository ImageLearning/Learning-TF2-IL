import urllib.request
from html.parser import HTMLParser



var_url = "https://octodex.github.com"
var_Target_folder =".//octocats/"


def SetupImage(Image_URL):
    New_Image_URL : str = Image_URL
    ToDownload : bool = False
    Target_Name : str = ""
    if(Image_URL[0:5] == "https"):
        New_Image_URL = Image_URL
        ToDownload = False
    elif(False):
        New_Image_URL = Image_URL
        ToDownload = False
    else:
        New_Image_URL = var_url + Image_URL
        Target_Name = Image_URL[8:len(Image_URL)]
        ToDownload = True

    if(ToDownload == True):
        print(New_Image_URL)
        RetrieveImage(New_Image_URL,Target_Name)


def RetrieveImage(arg_url, arg_filename):
    urllib.request.urlretrieve(arg_url, var_Target_folder + arg_filename )
    return arg_filename


class HTMLImageAndLinkParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        if tag == "img":
            for name, value in attrs:
                if(name == "src"):
                    SetupImage(value)
                elif(name == "data-src"):
                    SetupImage(value)
                #else:
                #    print(name + " : " + value)



WebPage = urllib.request.urlopen(var_url)
#print(WebPage)
#print("WebPage acquired \r\n")
WebPageData: str = WebPage.read()
#print(WebPageData)
#print("WebPage Read \r\n")
FindOctoKitties = HTMLImageAndLinkParser()
FindOctoKitties.reset()
#print(" Parser Created\r\n")
#print(type(WebPageData))
FindOctoKitties.feed(WebPageData.decode("utf-8"))





