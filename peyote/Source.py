class Source:

    def __init__(self, name):
        self.name = name


class Glitch(Source):
    def __init__(self, name):
        Source.__init__(name)


class AstrophysicalSource(Source):
    def __init__(self, name):
        Source.__init__(name)


class CompactBinaryCoalescence(AstrophysicalSource):
    def __init__(self, name):
        AstrophysicalSource.__init__(name)


class Supernova(AstrophysicalSource):
    def __init__(self, name):
        AstrophysicalSource.__init__(name)


class BinaryBlackHole(CompactBinaryCoalescence):
    def __init__(self, name):
        CompactBinaryCoalescence.__init__(name)


class BinaryBlackNeutronStar(CompactBinaryCoalescence):
    def __init__(self, name):
        CompactBinaryCoalescence.__init__(name)
