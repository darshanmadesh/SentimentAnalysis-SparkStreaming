import re, string, unicodedata
import contractions
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet

# download these once if not present
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')

# removing emojis
def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

#remove urls
def remove_url(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\S+', '', text)
    #remove tagged usernames
    text = re.sub(r'@\S+', '', text)
    return text

# remove non-ascii characters
def remove_non_ascii(text):
    words = []
    for word in text.split():
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        words.append(new_word)
    return " ".join(words)

# handle punctuation by adding whitespaces
regular_punct = list(string.punctuation)
extra_punct = [
    ',', '.', '"', ':', ')', '(', '!', '?', '|', ';', "'", '$', '&',
    '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
    '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',
    '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”',
    '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾',
    '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼',
    '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',
    'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»',
    '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø',
    '¹', '≤', '‡', '√', '«', '»', '´', 'º', '¾', '¡', '§', '£', '₤']
all_punct = list(set(regular_punct + extra_punct))
def spacing_punctuation(text):
    for punc in all_punct:
        if punc in text:
            text = text.replace(punc, f'')
    return text

def cleanData(text):
    text = remove_emoji(text)
    text = remove_url(text)
    text = contractions.fix(text)
    text = remove_non_ascii(text)
    text = spacing_punctuation(text)
    text = text.lower()
    return text

lemmatizer = WordNetLemmatizer()

def pos_tagger(word):
	tag = nltk.pos_tag([word])[0][1][0].upper()
	tag_dict = {
		"J" : wordnet.ADJ,
		"N" : wordnet.NOUN,
		"V" : wordnet.VERB,
		"R" : wordnet.ADV
	}
	return tag_dict.get(tag, wordnet.NOUN)

def lemmatizeData(text):
	# first clean the data
	text = cleanData(text)
	words = nltk.word_tokenize(text)
	words = [lemmatizer.lemmatize(word, pos_tagger(word)) for word in words if word not in set(stopwords.words('english'))]
	return " ".join(words)
	
