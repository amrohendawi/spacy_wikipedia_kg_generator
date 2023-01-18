import pandas as pd
import spacy
import neuralcoref

import re


class EntityExtractor:
    def __init__(self):
        spacy.prefer_gpu()
        self.nlp = spacy.load("en_core_web_lg")
        neuralcoref.add_to_pipe(self.nlp)

    def preprocess_text(self, text):
        text = re.sub(r'\n+', '.', text)  # replace multiple newlines with period
        text = re.sub(r'\[\d+\]', ' ', text)  # remove reference numbers
        return text

    def filter_spans(self, spans):
        # Filter a sequence of spans so they don't contain overlaps
        # For spaCy 2.1.4+: this function is available as spacy.util.filter_spans()
        get_sort_key = lambda span: (span.end - span.start, -span.start)
        sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
        result = []
        seen_tokens = set()
        for span in sorted_spans:
            # Check for end - 1 here because boundaries are inclusive
            if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
                result.append(span)
            seen_tokens.update(range(span.start, span.end))
        result = sorted(result, key=lambda span: span.start)
        return result

    def refine_ent(self, ent, sent):
        unwanted_tokens = (
            'PRON',  # pronouns
            'PART',  # particle
            'DET',  # determiner
            'SCONJ',  # subordinating conjunction
            'PUNCT',  # punctuation
            'SYM',  # symbol
            'X',  # other
        )
        ent_type = ent.ent_type_  # get entity type
        if ent_type == '':
            ent_type = 'NOUN_CHUNK'
            ent = ' '.join(str(t.text) for t in
                           self.nlp(str(ent)) if t.pos_
                           not in unwanted_tokens and t.is_stop == False)
        elif ent_type in ('NOMINAL', 'CARDINAL', 'ORDINAL') and str(ent).find(' ') == -1:
            refined = ''
            for i in range(len(sent) - ent.i):
                if ent.nbor(i).pos_ not in ('VERB', 'PUNCT'):
                    refined += ' ' + str(ent.nbor(i))
                else:
                    ent = refined.strip()
                    break

        return ent, ent_type

    def extract_entities(self, text, coref=True):
        text = self.nlp(text)
        if coref:
            text = self.nlp(text._.coref_resolved)  # resolve coreference clusters
        sentences = [sent.string.strip() for sent in text.sents]  # split text into sentences
        print("starting entity extraction")
        ent_pairs = []
        for sent in sentences:
            sent = self.nlp(sent)
            spans = list(sent.ents) + list(sent.noun_chunks)  # collect nodes
            spans = self.filter_spans(spans)
            with sent.retokenize() as retokenizer:
                [retokenizer.merge(span, attrs={'tag': span.root.tag,
                                                'dep': span.root.dep}) for span in spans]
            deps = [token.dep_ for token in sent]

            # limit our example to simple sentences with one subject and object
            if (deps.count('obj') + deps.count('dobj')) != 1 \
                    or (deps.count('subj') + deps.count('nsubj')) != 1:
                continue

            for token in sent:
                if token.dep_ not in ('obj', 'dobj'):  # identify object nodes
                    continue
                subject = [w for w in token.head.lefts if w.dep_
                           in ('subj', 'nsubj')]  # identify subject nodes
                if subject:
                    subject = subject[0]
                    # identify relationship by root dependency
                    relation = [w for w in token.ancestors if w.dep_ == 'ROOT']
                    if relation:
                        relation = relation[0]
                        # add adposition or particle to relationship
                        if relation.nbor(1).pos_ in ('ADP', 'PART'):
                            relation = ' '.join(
                                (str(relation), str(relation.nbor(1))))
                    else:
                        relation = 'unknown'

                    subject, subject_type = self.refine_ent(subject, sent)
                    token, object_type = self.refine_ent(token, sent)

                    ent_pairs.append([str(subject), str(relation), str(token),
                                      str(subject_type), str(object_type)])

        ent_pairs = [sublist for sublist in ent_pairs
                     if not any(str(ent) == '' for ent in sublist)]
        pairs = pd.DataFrame(ent_pairs, columns=['subject', 'relation', 'object',
                                                 'subject_type', 'object_type'])
        print('Entity pairs extracted:', str(len(ent_pairs)))
        return pairs

    def get_entity_pairs(self, text, coref=True):
        preprocessed_text = self.preprocess_text(text)
        return self.extract_entities(preprocessed_text, coref)
