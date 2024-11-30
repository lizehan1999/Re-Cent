import re


class TokenSplitterBase():
    def __init__(self):
        pass

    def __call__(self, text) -> (str, int, int):
        pass


class WhitespaceTokenSplitter(TokenSplitterBase):
    def __init__(self):
        self.whitespace_pattern = re.compile(r'\w+(?:[-_]\w+)*|\S')

    def __call__(self, text):
        for match in self.whitespace_pattern.finditer(text):
            yield match.group(), match.start(), match.end()


class WordsSplitter(TokenSplitterBase):
    def __init__(self):
        self.splitter = WhitespaceTokenSplitter()

    def __call__(self, text):
        for token in self.splitter(text):
            yield token
