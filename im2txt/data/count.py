from collections import Counter
counter = Counter()
captions=[['<S>', 'black', 'and', 'yellow', 'umbrellas', 'open', 'on', 'the', 'grass', 'of', 'a', 'park', '.', '</S>'],
         ['<S>', 'a', 'lush', 'green', 'field', 'full', 'of', 'yellow', 'umbrellas', '.', '</S>']]
for c in captions:
	counter.update(c)
print(counter)
