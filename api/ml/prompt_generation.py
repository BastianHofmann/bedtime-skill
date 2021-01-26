import random
import itertools
from model_loader import ModelLoader

model = ModelLoader("NY4", "http://ec2-18-193-73-211.eu-central-1.compute.amazonaws.com:8000").get_model()

one_prompts = [
    lambda a: f'Once up a time a {a}',
    lambda a: f'At the time when a {a}',
    lambda a: f'A long time back a {a}',
    lambda a: f'In a certain village there was a {a}',
    lambda a: f'In a certain land there was a {a}',
    lambda a: f'Once there was a {a}',
    lambda a: f'Once a {a}',
    lambda a: f'This is a story about a {a}',
    lambda a: f'Somewhere or other, but I don\'t know where, there was a {a}',
]

two_prompts = [
    lambda a,b: f'Once up a time a {a} and a {b}',
    lambda a,b: f'At the time when a {a} and a {b}',
    lambda a,b: f'A long time back a {a} and a {b}',
    lambda a,b: f'In a certain village there was a {a} and a {b}',
    lambda a,b: f'In a certain land there was a {a} and a {b}',
    lambda a,b: f'Once there was a {a} and a {b}',
    lambda a,b: f'Once a {a} and a {b}',
    lambda a,b: f'This is a story about a {a} and a {b}',
    lambda a,b: f'Somewhere or other, but I don\'t know where, there was a {a} and a {b}',
]

three_prompts = [
    lambda a,b,c: f'Once up a time a {a}, a {b} and a {c}',
    lambda a,b,c: f'At the time when a {a}, a {b} and a {c}',
    lambda a,b,c: f'A long time back a {a}, a {b} and a {c}',
    lambda a,b,c: f'In a certain village there was a {a}, a {b} and a {c}',
    lambda a,b,c: f'In a certain land there was a {a}, a {b} and a {c}',
    lambda a,b,c: f'Once there was a {a}, a {b} and a {c}',
    lambda a,b,c: f'Once a {a}, a {b} and a {c}',
    lambda a,b,c: f'This is a story about a {a}, a {b} and a {c}',
    lambda a,b,c: f'Somewhere or other, but I don\'t know where, there was a {a}, a {b} and a {c}',
]

keywords = ['cat', 'dog', 'horse', 'mouse', 'dragon', 'king', 'queen', 'princess', 'prince']

one_stories = [(a,model.predict(random.choice(one_prompts)(keyword), max_length=300)) for a in keywords]
once_cut_stories = [(a,story[0:story.rfind('.')+1]) for a,story in one_stories]

two_stories = [(a,b,model.predict(random.choice(two_prompts)(a,b), max_length=300)) for a,b in itertools.product(keywords, keywords) if a != b]
two_cut_stories = [(a,b,story[0:story.rfind('.')+1]) for a,b,story in two_stories]

three_stories = [(a,b,c,model.predict(random.choice(three_prompts)(a,b,c), max_length=300)) for a,b,c in itertools.product(keywords, keywords, keywords) if a != b and b != c and a != c]
three_cut_stories = [(a,b,c,story[0:story.rfind('.')+1]) for a,b,c,story in three_stories]

for a, story in one_stores:
    model.upload_prediction(a, story)

for a, b, story in one_stores:
    model.upload_prediction(a, story, b)

for a, b, c, story in one_stores:
    model.upload_prediction(a, story, b, c)
