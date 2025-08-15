from typing import List
import re
# import pkuseg
import numpy as np
# from utils import SplitSentence

# segmenter = pkuseg.pkuseg()


class Node:
    text: str
    beam_size: int = 3
    max_depth: int = 10
    finished: bool = False
    children: List['Node']
    negative_prefix: List[str]
    ancestors: List['Node']
    PATTERN_SPLITER = '<|pattern|>'

    def __init__(self, text, lang='zh',
                 generate_fn=None, reward_fn=None,
                 max_token=150, beam_size=3,
                 default_prefix=None, default_negative_prefix=None):
        self.text = text
        self.lang = lang
        self.children = []
        self.ancestors = []
        self.negative_prefix = []
        if default_negative_prefix is not None:
            assert isinstance(default_negative_prefix, list)
            self.default_negative_prefix = default_negative_prefix
            self.negative_prefix = default_negative_prefix
        self.generate_fn = generate_fn
        self.reward_fn = reward_fn
        self.max_token = max_token
        self.beam_size = beam_size
        self.default_prefix = default_prefix

        if self.lang == 'zh':
            self.sentence_spliter = SplitSentence()
        elif self.lang == 'en':
            from pysbd import Segmenter
            self.sentence_spliter = Segmenter(language='en', clean=False).segment

    def __repr__(self):
        _prompt = ''
        for node in self.ancestors:
            _prompt += f'{node.text}'
        pattern = ''.join([f'(?!{re.escape(p)})' for p in self.negative_prefix])
        if self.lang == 'en':
            pattern = r'\s' + pattern
        text = self.text
        if self.default_prefix is not None:
            text += self.default_prefix
        return _prompt + text + self.PATTERN_SPLITER + pattern + '.*'

    def get_user_query(self):
        if len(self.ancestors) == 0:
            query = self.text
        else:
            query = self.ancestors[0].text
        query = query.strip('<|im_start|>user\n').split('<|im_end|>\n<|im_start|>assistant')[0]
        return query

    def segment(self, text) -> list:
        if self.lang == 'zh':
            raise NotImplementedError
            # return segmenter.cut(text)
        elif self.lang == 'en':
            return text.split(' ')
        else:
            raise NotImplementedError

    def add_neg(self, sent):
        # print('add_neg', sent)
        sent = sent.strip()
        negative_prefix = self.segment(sent)[:3]
        if self.lang == 'zh':
            negative_prefix = ''.join(negative_prefix)
        elif self.lang == 'en':
            negative_prefix = ' '.join(negative_prefix)
        # if len(negative_prefix) > 8:
        #     negative_prefix = negative_prefix[:8]
        if negative_prefix not in self.negative_prefix:
            self.negative_prefix.append(negative_prefix)
        return self.negative_prefix

    def add_pos(self, sent):
        new_node = Node(sent,
                        reward_fn=self.reward_fn, generate_fn=self.generate_fn,
                        max_token=self.max_token, beam_size=self.beam_size,
                        default_negative_prefix=self.negative_prefix)
        new_node.ancestors = self.ancestors + [self]
        self.children.append(new_node)
        return new_node

    def run(self):
        if self.finished or len(self.ancestors) >= self.max_depth:
            return

        for _ in range(40):
            prompt, pattern = self.__repr__().split(self.PATTERN_SPLITER)
            # print('prompt ======')
            # print(prompt)
            # print('========')
            choices = self.generate_fn(prompt, pattern, n=5)
            # print(choices)
            for choice in choices:
                if len(self.children) >= self.beam_size:
                    self.finished = True
                    return
                new_sent = self.sentence_spliter(choice.text)[0]
                context = self.__repr__().split(self.PATTERN_SPLITER)[0]
                is_safety, _ = self.reward_fn(self.get_user_query(), context, new_sent)
                if is_safety:
                    self.add_neg(new_sent)
                    if len(self.negative_prefix) > 20:
                        # Too many negative patterns, abort this node (path).
                        self.finished = True
                        return
                else:
                    if new_sent in [child.text for child in self.children]:
                        _node = self.add_pos(new_sent)
                        _node.finished = True  # The new node is already in the tree, skip it
                    else:
                        _node = self.add_pos(new_sent)
                        if len(_node.__repr__().split(self.PATTERN_SPLITER)[0]) - len(
                                self.get_user_query()) > self.max_token:
                            # Reach max token, stop sampling
                            self.finished = True
                            return
                        else:
                            _node.run()

        for child in self.children:
            if not child.finished:
                child.run()

        self.finished = True


class AttackPrefixTree:
    tree: Node = None

    def __init__(self, generate_fn=None, max_token=150, default_prefix=None, default_negative_prefix=None,
                 reward_fn=None, lang='zh'):
        self.max_token = max_token
        self.generate_fn = generate_fn
        self.reward_fn = reward_fn
        self.default_negative_prefix = default_negative_prefix
        self.lang = lang
        self.default_prefix = default_prefix

    def generate_unsafe(self, prompt):
        self.tree = Node(prompt, generate_fn=self.generate_fn, reward_fn=self.reward_fn, max_token=self.max_token,
                         default_negative_prefix=self.default_negative_prefix, lang=self.lang,
                         default_prefix=self.default_prefix)
        self.tree.run()

    def gather_all_leaves(self):
        leaves = []

        def _gather(node):
            if node.children is None or len(node.children) == 0:
                leaves.append(node)
            else:
                for child in node.children:
                    _gather(child)

        _gather(self.tree)
        return leaves

    def gather_all_paths(self):
        leaves = self.gather_all_leaves()
        paths = []
        for leaf in leaves:
            path = leaf.__repr__()
            paths.append(path)
        return paths

    def visualize(self):
        def print_tree(node, depth=0):
            print('\t' * depth + node.text)
            for child in node.children:
                print_tree(child, depth + 1)

        print_tree(self.tree)

    def reranking(self, k=1):
        paths = self.gather_all_paths()
        root = self.tree
        scores = []
        answers = []
        for path in paths:
            answer = path.split(self.tree.PATTERN_SPLITER)[0]
            answers.append(answer)
            is_safety, safety_score = self.reward_fn(root.get_user_query(), '', answer)
            if is_safety:
                scores.append(safety_score)
            else:
                scores.append(-safety_score)

        answers = np.array(answers)
        scores = np.array(scores)
        # return topk
        indices = np.argsort(scores)[::-1]
        return answers[indices][:k]
