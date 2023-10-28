from collections import Counter
import numpy as np
import math

"""
CS 4/6120, Fall 2023
Homework 3 - starter code
"""

# constants
SENTENCE_BEGIN = "<s>"
SENTENCE_END = "</s>"
UNK = "<UNK>"



# UTILITY FUNCTIONS

def create_ngrams(tokens: list, n: int) -> list:
  """Creates n-grams for the given token sequence.
  Args:
    tokens (list): a list of tokens as strings
    n (int): the length of n-grams to create

  Returns:
    list: list of tuples of strings, each tuple being one of the individual n-grams
  """
  # STUDENTS IMPLEMENT
  ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
  return ngrams
UNK_generation=False
def read_file(path: str) -> list:
  """
  Reads the contents of a file in line by line.
  Args:
    path (str): the location of the file to read

  Returns:
    list: list of strings, the contents of the file
  """
  if path=="training_files/unknowns.txt":
        global UNK_generation
        UNK_generation=True
     
  # PROVIDED
  f = open(path, "r", encoding="utf-8")
  contents = f.readlines()
  f.close()
  return contents

def tokenize_line(line: str, ngram: int, 
                   by_char: bool = True, 
                   sentence_begin: str=SENTENCE_BEGIN, 
                   sentence_end: str=SENTENCE_END):
  """
  Tokenize a single string. Glue on the appropriate number of 
  sentence begin tokens and sentence end tokens (ngram - 1), except
  for the case when ngram == 1, when there will be one sentence begin
  and one sentence end token.
  Args:
    line (str): text to tokenize
    ngram (int): ngram preparation number
    by_char (bool): default value True, if True, tokenize by character, if
      False, tokenize by whitespace
    sentence_begin (str): sentence begin token value
    sentence_end (str): sentence end token value

  Returns:
    list of strings - a single line tokenized
  """
  # PROVIDED
  inner_pieces = None
  if by_char:
    inner_pieces = list(line)
  else:
    # otherwise split on white space
    inner_pieces = line.split()

  if ngram == 1:
    tokens = [sentence_begin] + inner_pieces + [sentence_end]
  else:
    tokens = ([sentence_begin] * (ngram - 1)) + inner_pieces + ([sentence_end] * (ngram - 1))
  # always count the unigrams
  return tokens


def tokenize(data: list, ngram: int, 
                   by_char: bool = True, 
                   sentence_begin: str=SENTENCE_BEGIN, 
                   sentence_end: str=SENTENCE_END):
  """
  Tokenize each line in a list of strings. Glue on the appropriate number of 
  sentence begin tokens and sentence end tokens (ngram - 1), except
  for the case when ngram == 1, when there will be one sentence begin
  and one sentence end token.
  Args:
    data (list): list of strings to tokenize
    ngram (int): ngram preparation number
    by_char (bool): default value True, if True, tokenize by character, if
      False, tokenize by whitespace
    sentence_begin (str): sentence begin token value
    sentence_end (str): sentence end token value

  Returns:
    list of strings - all lines tokenized as one large list
  """
  # PROVIDED
  total = []
  # also glue on sentence begin and end items
  for line in data:
    line = line.strip()
    # skip empty lines
    if len(line) == 0:
      continue
    tokens = tokenize_line(line, ngram, by_char, sentence_begin, sentence_end)
    total += tokens
  return total


class LanguageModel:

  def __init__(self, n_gram):
    """Initializes an untrained LanguageModel
    Args:
      n_gram (int): the n-gram order of the language model to create
    """
    # STUDENTS IMPLEMENT
    self.n_gram = n_gram
    self.ngrams_count = Counter()
    self.context_count = Counter()
    self.vocab_length = 0
    self.no_of_characters = 1
    self.vocab = []
    self.total_tokens = 0
  
  def train(self, tokens: list, verbose: bool = False) -> None:
    """Trains the language model on the given data. Assumes that the given data
    has tokens that are white-space separated, has one sentence per line, and
    that the sentences begin with <s> and end with </s>
    Args:
      tokens (list): tokenized data to be trained on as a single list
      verbose (bool): default value False, to be used to turn on/off debugging prints
    """
    # STUDENTS IMPLEMENT
    self.vocab = [token for token in tokens if token not in ('<s>', '</s>')]
    count_of_unigrams = Counter([token for token in tokens if token not in ('<s>', '</s>')])
    self.no_of_characters = sum(count_of_unigrams.values())
    self.vocab_length = len(set(tokens))
    self.total_tokens = len(tokens)

    ngrams = create_ngrams(tokens, self.n_gram)
    self.ngrams_count.update(ngrams)
    for ngram in ngrams:
        context = tuple(ngram[:-1])
        self.context_count[context] += 1

    if verbose:
        print("Training completed.")

    """
    calculate_probability function calculates all the probabilites for unigrams bigrams and trigrams  
    It is then passed to the score function via a loop so that its calculated for each and every ngram 
    """
  def calculate_probability(self, ngram: tuple, context: tuple) -> float:
    ngram_count = self.ngrams_count[ngram]
    context_count = self.context_count[context] if context in self.context_count else 0

        # Laplace smoothing with +1 in the numerator
    if (self.n_gram == 1) and ngram not in self.ngrams_count.keys():
        probability = (ngram_count + 2) / (self.total_tokens + self.vocab_length)
    elif self.n_gram == 1:
        probability = (ngram_count + 1) / (self.total_tokens + self.vocab_length)
    else:
        probability = (ngram_count + 1) / (context_count + self.vocab_length)

    return probability



  def score(self, sentence_tokens: list) -> float:
    """Calculates the probability score for a given string representing a single sequence of tokens.
    Args:
      sentence_tokens (list): a tokenized sequence to be scored by this model
      
    Returns:
      float: the probability value of the given tokens for this model
    """
    # STUDENTS IMPLEMENT
    ngrams = create_ngrams(sentence_tokens, self.n_gram)
    score = 1.0
    for ngram in ngrams:
        context = tuple(ngram[:-1])
        probability = self.calculate_probability(ngram, context)
        score *= probability
    return score

  def generate_sentence(self) -> list:
    """Generates a single sentence from a trained language model using the Shannon technique.
      
    Returns:
      list: the generated sentence as a list of tokens
    """
    # STUDENTS IMPLEMENT
    generated_sentence = [SENTENCE_BEGIN]
    next_token = UNK  
    max_iterations = 1000 

    for _ in range(max_iterations):
        context = tuple(generated_sentence[-(self.n_gram - 1):])

        if self.n_gram == 1:
            possible_next_tokens = [ngram[0] for ngram in self.ngrams_count.keys()]
        else:
            possible_next_tokens = [ngram[-1] for ngram in self.ngrams_count.keys() if ngram[:-1] == context]

        if not possible_next_tokens or all(token == UNK for token in possible_next_tokens) or UNK_generation==True:
            next_token = UNK
        else:
            #This checks the probability of each next occuring token based on previous tokens i..e context
            probabilities = [self.calculate_probability((token,), context) for token in possible_next_tokens]
            #this is for normalizing the probabilites so that a token can be choosen based on it
            probabilities = np.array(probabilities) / sum(probabilities)
            #This samples a radnom token with the highest probability
            next_token = np.random.choice(possible_next_tokens, p=probabilities)

        generated_sentence.append(next_token)

        if next_token == SENTENCE_END:
            break
        if next_token is None:
            generated_sentence.append(SENTENCE_END)

    return generated_sentence


  def generate(self, n: int) -> list:
    """Generates n sentences from a trained language model using the Shannon technique.
    Args:
      n (int): the number of sentences to generate
      v
    Returns:
      list: a list containing lists of strings, one per generated sentence
    """
    # PROVIDED
    return [self.generate_sentence() for _ in range(n)]


  def perplexity(self, sequence: list) -> float:
    """Calculates the perplexity score for a given sequence of tokens.
    Args:
      sequence (list): a tokenized sequence to be evaluated for perplexity by this model
      
    Returns:
      float: the perplexity value of the given sequence for this model
    """
    # 6120 IMPLEMENTS
    ngrams = create_ngrams(sequence, self.n_gram)
    log_probability = 0.0
    for ngram in ngrams:
        context = tuple(ngram[:-1])
        # We are first calculating the sum of all the probabilites in that sentence for a paticular n_gram
        probability = self.calculate_probability(ngram, context)
        #Then we are logging the probability
        log_probability += math.log(probability)
    #After everything is calculated we then plugin the values into the formula to get the required preplexity
    perplexity = math.exp(-log_probability / len(ngrams))
    return perplexity
  
# not required
if __name__ == '__main__':
    print('')