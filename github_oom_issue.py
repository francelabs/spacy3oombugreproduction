import os
import spacy
import torch
from thinc.api import set_gpu_allocator, require_gpu
import cupy
from typing import List, Dict, Any, Tuple
from bisect import bisect_left
from spacy.tokens import Doc


ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/'
# the directory where to save files
source = ROOT_DIR + 'files/'
# The maximum number of characters of each block of text
MAX_CHARS = 100000

# Load the small model, it is only used to separate sentences without getting oom errors that would occur with the transformer
SENTER = spacy.load("en_core_web_sm", exclude=['tok2vec', 'tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])
SENTER.enable_pipe("senter")

set_gpu_allocator("pytorch")
require_gpu(0)


########## HELPER METHODS TO CUT DOCUMENTS INTO SMALLER BLOCKS #########

# The following methods are not really interesting for the issue. I am just dividing documents
# into blocks of at most MAX_CHARS characters.
def split_long_sentences(doc: Doc, sent_boundaries: List[int]):
    """Go through sentence boundaries and check if there are single sentences longer than MAX_CHARS.
    Split them to the nearest token, if possible. If the token itself is longer than MAX_CHARS it is
    divided."""
    token_bounds = [tok.idx + len(tok.text) for tok in doc]  # character offset after end of each token
    token_bounds.insert(0, 0)
    idx = 0
    tok_idx = 0
    while idx < len(sent_boundaries) - 1:
        if sent_boundaries[idx + 1] - sent_boundaries[idx] > MAX_CHARS:
            # print(f"boundaries: {sent_boundaries[idx+1]}  prev: {sent_boundaries[idx]}")
            to_find = sent_boundaries[idx] + MAX_CHARS
            tok_idx = bisect_left(token_bounds, to_find, lo=tok_idx)
            if token_bounds[tok_idx] == to_find:  # MAX_CHARS at exactly end of token
                sent_boundaries.insert(idx + 1, to_find)
            elif token_bounds[tok_idx - 1] > sent_boundaries[idx]:
                sent_boundaries.insert(idx + 1, token_bounds[tok_idx - 1])
            else:  # token is too long, i have to cut the token
                sent_boundaries.insert(idx + 1, to_find)
        idx += 1


def split_document(text: str) -> Tuple[List[str], List[int]]:
    """Split the document in blocks of text of size at most MAX_CHARS.
    Returns the blocks of text and for each block its offset from the start of the document."""
    texts = []
    offsets = [0]
    doc = SENTER(text)
    sents = list(doc.sents)
    sent_boundaries = [sent.end_char for sent in sents]
    sent_boundaries.insert(0, 0)
    split_long_sentences(doc, sent_boundaries)
    start_char = 0
    idx = 0  # the index of the last sentence that was already added
    while start_char + MAX_CHARS < len(text):
        to_find = start_char + MAX_CHARS
        new_idx = bisect_left(sent_boundaries, to_find, lo=idx) - 1  # bisect returns the index of the first >=
        if new_idx == idx:  # the block has exactly MAX_CHAR characters
            assert sent_boundaries[idx + 1] - sent_boundaries[idx] == MAX_CHARS
            new_idx += 1
        idx = new_idx
        end_char = sent_boundaries[idx]
        texts.append(text[start_char: end_char])
        start_char = end_char
        offsets.append(start_char)

    texts.append(text[start_char:])
    return texts, offsets


########## END OF HELPER METHODS ##########


def get_entities(doc: Doc, offset: int) -> List[Dict[str, Any]]:
    """Extract the data to return from the REST API given a Doc object. Modify
    this function to include other data."""
    ents = [
        {
            "label": ent.label_,
            "start": ent.start_char + offset,
            "end": ent.end_char + offset,
        }
        for ent in doc.ents
    ]
    return ents


# A generator to get content of files from file names
def texts_generator(file_names):
    for file_name in file_names:
        with open(source + file_name, 'r') as f:
            content = f.read()
            yield content


def main():
    """Iterate over documents, split them into blocks of at most MAX_CHARS characters, extract entities
    and combine them"""
    file_names = ["Enron%20Bonus%20List.txt", "Florida%20Energy%20Market.txt", "Goldman%20Sachs%20Presentation.txt"]

    nlp = spacy.load("en_core_web_trf")
    for pipe in ["tagger", "parser", "attribute_ruler", "lemmatizer"]:
        nlp.disable_pipe(pipe)
    nlp.get_pipe("transformer").model.attrs["flush_cache_chance"] = 1

    texts = texts_generator(file_names)

    for idx, text in enumerate(texts):
        print(file_names[idx] + ' ' + str(os.path.getsize(source + file_names[idx]) / 1000) + ' kB')
        response_body = []
        blocks, offsets = split_document(text)
        count = 0
        for t, offset in zip(blocks, offsets):
            count += 1

            torch.cuda.empty_cache()
            cupy.get_default_memory_pool().free_all_blocks()

            print(f"block {count}\t\toffset: {offset}")
            doc = nlp(t)
            response_body += get_entities(doc, offset)
            doc._.trf_data = None   # useless, but just to be sure


if __name__ == "__main__":
    main()
