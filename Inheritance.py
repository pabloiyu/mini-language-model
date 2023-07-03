"""
I will try to implement the concept of parent class and child class. Furthermore, I will also implement the concept
of nested classes.
"""

class Dog():
    def __init__(self, name):
        self.name = name

    def bark(self):
        print("bark")


class Labrador(Dog):
    def __init__(self, name):
        super.__init__(name)
        self.age = 2

    def speak(self):
        print("intelligent dog")

dog = Labrador("Perry")

# We should now be able to access the name attribue and bark function from the object instantiated from the child class.abs
print(dog.name)
dog.bark()
dog.speak()