# Copyright (c) 2017-present, Moscow Institute of Physics and Technology.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.dialog_teacher import DialogTeacher


class DefaultTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        self.data_path = DefaultTeacher._path(opt)
        opt['datafile'] = self.data_path
        self.id = 'UtteranceQE'
        self.dialogs = None
        super().__init__(opt, shared)

    @staticmethod
    def _path(opt):
        import os
        import sys
        from parlai.tasks.utterance_qe.build import build
        build(opt)
        dt = opt['datatype'].split(':')[0]

        if dt == 'train':
            path = os.path.join(opt['datapath'], 'UtteranceQE', 'train.json')
        elif dt == 'test':
            path = os.path.join(opt['datapath'], 'UtteranceQE', 'test.json')
        elif dt == 'valid':
            print('warning: validation is not supporting', file=sys.stderr)
            path = None
        else:
            raise RuntimeError('Not valid datatype.')

        return path

    @staticmethod
    def _transform_utterance(utterance, user_types):
        uid = utterance['userId']
        t = user_types[uid]
        eval = '?'
        if utterance['evaluation'] == 1:
            eval = 'dislike'
        elif utterance['evaluation'] == 2:
            eval = 'like'
        return ': '.join([utterance['userId'] + '(' + t + ')', utterance['text']]), eval

    def setup_data(self, path):
        import json
        print('loading: ' + path)

        if path is None:
            return iter(())

        with open(path) as data_file:
            self.dialogs = json.load(data_file)

        for dialog in self.dialogs:
            user_types = dict(map(lambda u: (u['id'], u['userType']), dialog['users']))
            threads_evals = [i for i in map(lambda u: DefaultTeacher._transform_utterance(u, user_types),
                                          dialog["thread"])]
            for i, (utterance, eval) in enumerate(threads_evals):
                episode_done = False
                if i == len(dialog["thread"]) - 1:
                    episode_done = True

                yield (utterance, [eval]), episode_done
