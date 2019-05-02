class SubclassPropertyAccessor(object):
    """
    Generic descriptor class that allows handy access of properties without long
    boilerplate code. The properties of Interferometer are defined as instances
    of this class. Don't touch this if you don't know what you are doing.

    This avoids lengthy code like
    ```
    @property
    def length(self):
        return self.geometry.length

    @length_setter
    def length(self, length)
        self.geometry.length = length

    in the Interferometer class
    ```
    """

    def __init__(self, property_name, container_instance_name):
        self.property_name = property_name
        self.container_instance_name = container_instance_name

    def __get__(self, instance, owner):
        return getattr(getattr(instance, self.container_instance_name), self.property_name)

    def __set__(self, instance, value):
        setattr(getattr(instance, self.container_instance_name), self.property_name, value)
