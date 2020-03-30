

# https://www.w3schools.com/python/python_intro.asp

# https://docs.python.org/3/

def add(a, b):
    return a + b

def ForEachLoop():
    for letter in 'Python':     # First Example
        print( 'Current Letter :', letter ) # ( ) are optional, sometimes

def IsTrue(value):
    return bool(value) #No indentation

def function():
    print("inside a function that returns nothing")

def main():
    function()
    ForEachLoop()
    print( add(2,2) )


main()