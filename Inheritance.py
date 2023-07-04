"""
I will try to implement the concept of parent class and child class. Furthermore, I will also implement the concept
of nested classes which inherit from each other.
"""

# You don't need to  
class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def bark(self):
        print("bark")

    def say_name(self):
        print(self.name)

    def say_age(self):
        print(self.age)


class Labrador(Dog):
    def __init__(self, name, age):
        super().__init__(name, age)

    def speak(self):
        print("intelligent dog")

dog = Labrador("Perry", 10)

# We should now be able to access the name attribute and bark function from the object instantiated from the child class.
print(dog.name)
dog.bark()
dog.speak()
dog.say_name()
dog.say_age()
print("##########################")


######################################################################################################################################
# Let's now try to define nested classes which inherit from each other

class Configuration:
    start = "initialise"

    def __init__(self, purpose):
        self.purpose = purpose

    # The env class is conceptually similar to a struct in C++. In this case, the class has no defined methods, and is essentially a
    # container to group together related attributes.
    class env:
        # When you define an attribute within a class without the "self." prefix, it becomes a class attribute. It exists independently 
        # of having to instantiate an object of that class. 
        num_envs = 4096
        num_actions = 12

# We can call the class attribute/variable without instantiating a Configuration object
print(Configuration.start)


class RobotConfiguration(Configuration):
    def __init__(self, purpose):
        super().__init__(purpose)

    # If we let the env class inherit from the Configuration.env class, then we can access class attributes defined in the parent class.
    # Otherwise, we would essentially overwrite the definition of the env class, making it a new and separate class within the 
    # RobotConfiguration class namespace.
    class env(Configuration.env):
        num_DOF = 10
    
conf = RobotConfiguration("Quadruped")

# We can also still call the "start" class attribute defined in the parent class
print(conf.start)
print(conf.env.num_actions)

