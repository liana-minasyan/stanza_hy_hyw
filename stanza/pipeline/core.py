"""
Pipeline that runs tokenize,mwt,pos,lemma,depparse
"""

import io
import itertools
import sys
import torch
import logging
import json
import os
import csv
import numpy as np
import statistics

import stanza.models.config as config
from stanza.utils.conll import CoNLL #liana
from distutils.util import strtobool
from stanza.pipeline._constants import *
from stanza.models.common.doc import Document
from stanza.pipeline.processor import Processor, ProcessorRequirementsException
from stanza.pipeline.registry import NAME_TO_PROCESSOR_CLASS, PIPELINE_NAMES
from stanza.pipeline.tokenize_processor import TokenizeProcessor
from stanza.pipeline.mwt_processor import MWTProcessor
from stanza.pipeline.pos_processor import POSProcessor
from stanza.pipeline.lemma_processor import LemmaProcessor
from stanza.pipeline.depparse_processor import DepparseProcessor
from stanza.pipeline.sentiment_processor import SentimentProcessor
from stanza.pipeline.ner_processor import NERProcessor
from stanza.resources.common import DEFAULT_MODEL_DIR, \
    maintain_processor_list, add_dependencies, build_default_config, set_logging_level, process_pipeline_parameters, sort_processors
from stanza.utils.helper_func import make_table

logger = logging.getLogger('stanza')

class ResourcesFileNotFoundError(FileNotFoundError):
    def __init__(self, resources_filepath):
        super().__init__(f"Resources file not found at: {resources_filepath}  Try to download the model again.")
        self.resources_filepath = resources_filepath

class LanguageNotDownloadedError(FileNotFoundError):
    def __init__(self, lang, lang_dir, model_path):
        super().__init__(f'Could not find the model file {model_path}.  The expected model directory {lang_dir} is missing.  Perhaps you need to run stanza.download("{lang}")')
        self.lang = lang
        self.lang_dir = lang_dir
        self.model_path = model_path

class UnsupportedProcessorError(FileNotFoundError):
    def __init__(self, processor, lang):
        super().__init__(f'Processor {processor} is not known for language {lang}.  If you have created your own model, please specify the {processor}_model_path parameter when creating the pipeline.')
        self.processor = processor
        self.lang = lang

class PipelineRequirementsException(Exception):
    """
    Exception indicating one or more requirements failures while attempting to build a pipeline.
    Contains a ProcessorRequirementsException list.
    """

    def __init__(self, processor_req_fails):
        self._processor_req_fails = processor_req_fails
        self.build_message()

    @property
    def processor_req_fails(self):
        return self._processor_req_fails

    def build_message(self):
        err_msg = io.StringIO()
        print(*[req_fail.message for req_fail in self.processor_req_fails], sep='\n', file=err_msg)
        self.message = '\n\n' + err_msg.getvalue()

    def __str__(self):
        return self.message


class Pipeline:

    def __init__(self, lang='en', dir=DEFAULT_MODEL_DIR, package='default', processors={}, logging_level=None, verbose=None, use_gpu=True, **kwargs):
        self.lang, self.dir, self.kwargs = lang, dir, kwargs

        # set global logging level
        set_logging_level(logging_level, verbose)
        print('hello from liana')
        # process different pipeline parameters
        lang, dir, package, processors = process_pipeline_parameters(lang, dir, package, processors)

        # Load resources.json to obtain latest packages.
        logger.debug('Loading resource file...')
        resources_filepath = os.path.join(dir, 'resources.json')
        if not os.path.exists(resources_filepath):
            raise ResourcesFileNotFoundError(resources_filepath)
        with open(resources_filepath) as infile:
            resources = json.load(infile)
        if lang in resources:
            if 'alias' in resources[lang]:
                logger.info(f'"{lang}" is an alias for "{resources[lang]["alias"]}"')
                lang = resources[lang]['alias']
            lang_name = resources[lang]['lang_name'] if 'lang_name' in resources[lang] else ''
        else:
            logger.warning(f'Unsupported language: {lang}.')

        # Maintain load list
        self.load_list = maintain_processor_list(resources, lang, package, processors) if lang in resources else []
        self.load_list = add_dependencies(resources, lang, self.load_list) if lang in resources else []
        self.load_list = self.update_kwargs(kwargs, self.load_list)
        if len(self.load_list) == 0:
            raise ValueError('No processors to load for language {}.  Please check if your language or package is correctly set.'.format(lang))
        load_table = make_table(['Processor', 'Package'], [row[:2] for row in self.load_list])
        logger.info(f'Loading these models for language: {lang} ({lang_name}):\n{load_table}')

        self.config = build_default_config(resources, lang, dir, self.load_list)
        self.config.update(kwargs)

        # Load processors
        self.processors = {}

        # configs that are the same for all processors
        pipeline_level_configs = {'lang': lang, 'mode': 'predict'}
        self.use_gpu = torch.cuda.is_available() and use_gpu
        logger.info("Use device: {}".format("gpu" if self.use_gpu else "cpu"))

        # set up processors
        pipeline_reqs_exceptions = []
        for item in self.load_list:
            processor_name, _, _ = item
            logger.info('Loading: ' + processor_name)
            curr_processor_config = self.filter_config(processor_name, self.config)
            curr_processor_config.update(pipeline_level_configs)
            logger.debug('With settings: ')
            logger.debug(curr_processor_config)
            try:
                # try to build processor, throw an exception if there is a requirements issue
                self.processors[processor_name] = NAME_TO_PROCESSOR_CLASS[processor_name](config=curr_processor_config,
                                                                                          pipeline=self,
                                                                                          use_gpu=self.use_gpu)
            except ProcessorRequirementsException as e:
                # if there was a requirements issue, add it to list which will be printed at end
                pipeline_reqs_exceptions.append(e)
                # add the broken processor to the loaded processors for the sake of analyzing the validity of the
                # entire proposed pipeline, but at this point the pipeline will not be built successfully
                self.processors[processor_name] = e.err_processor
            except FileNotFoundError as e:
                # For a FileNotFoundError, we try to guess if there's
                # a missing model directory or file.  If so, we
                # suggest the user try to download the models
                if 'model_path' in curr_processor_config:
                    model_path = curr_processor_config['model_path']
                    model_dir, model_name = os.path.split(model_path)
                    lang_dir = os.path.dirname(model_dir)
                    if not os.path.exists(lang_dir):
                        # model files for this language can't be found in the expected directory
                        raise LanguageNotDownloadedError(lang, lang_dir, model_path) from e
                    if processor_name not in resources[lang]:
                        # user asked for a model which doesn't exist for this language?
                        raise UnsupportedProcessorError(processor_name, lang)
                    if not os.path.exists(model_path):
                        model_name, _ = os.path.splitext(model_name)
                        # TODO: before recommending this, check that such a thing exists in resources.json.
                        # currently that case is handled by ignoring the model, anyway
                        raise FileNotFoundError('Could not find model file %s, although there are other models downloaded for language %s.  Perhaps you need to download a specific model.  Try: stanza.download(lang="%s",package=None,processors={"%s":"%s"})' % (model_path, lang, lang, processor_name, model_name)) from e

                # if we couldn't find a more suitable description of the
                # FileNotFoundError, just raise the old error
                raise

        # if there are any processor exceptions, throw an exception to indicate pipeline build failure
        if pipeline_reqs_exceptions:
            logger.info('\n')
            raise PipelineRequirementsException(pipeline_reqs_exceptions)

        logger.info("Done loading processors!")

    def update_kwargs(self, kwargs, processor_list):
        processor_dict = {processor: {'package': package, 'dependencies': dependencies} for (processor, package, dependencies) in processor_list}
        for key, value in kwargs.items():
            k, v = key.split('_', 1)
            if v == 'model_path':
                package = value if len(value) < 25 else value[:10]+ '...' + value[-10:]
                dependencies = processor_dict.get(k, {}).get('dependencies')
                processor_dict[k] = {'package': package, 'dependencies': dependencies}
        processor_list = [[processor, processor_dict[processor]['package'], processor_dict[processor]['dependencies']] for processor in processor_dict]
        processor_list = sort_processors(processor_list)
        return processor_list

    def filter_config(self, prefix, config_dict):
        filtered_dict = {}
        for key in config_dict.keys():
            k, v = key.split('_', 1) # split tokenize_pretokenize to tokenize+pretokenize
            if k == prefix:
                filtered_dict[v] = config_dict[key]
        return filtered_dict

    @property
    def loaded_processors(self):
        """
        Return all currently loaded processors in execution order.
        :return: list of Processor instances
        """
        return [self.processors[processor_name] for processor_name in PIPELINE_NAMES if self.processors.get(processor_name)]

    def process(self, doc):
        # run the pipeline

        # determine whether we are in bulk processing mode for multiple documents
        bulk=(isinstance(doc, list) and len(doc) > 0 and isinstance(doc[0], Document))
        for processor_name in PIPELINE_NAMES:
            if self.processors.get(processor_name):
                process = self.processors[processor_name].bulk_process if bulk else self.processors[processor_name].process
                doc = process(doc)
            doc2 = CoNLL.dict2conll(doc.to_dict(), 'data/test_pos_pipeline.conllu')
        # print('type of doc to dict', type(doc.to_dict()))
        # print('len doc_to_dict', len(doc.to_dict()))

        # scores = []
        # with open('stanza/pos_similar_score.csv') as csv_fnile:
        #     csv_reader = csv.reader(csv_file, delimiter=',')
        #     for row in csv_reader:
        #         scores.append(float(row[0]))
        # np_scores = np.asarray(scores)
        # med = statistics.median(scores)
        # threshold = np.percentile(scores, 85)
        # filtered_med = np_scores[(np_scores > med)]
        # filtered_med_indices = np.argwhere(np_scores > med)
        # filtered_threshold = np_scores[(np_scores > threshold)]
        # filtered_threshold_indices = np.argwhere(np_scores > threshold)
        # config.indices = filtered_med
        # config.indices_threshold = filtered_threshold
        # config.median = med
        # config.threshold = threshold
        # print('len filtered', len(filtered_med))
        # print('len threshold', len(filtered_threshold))
        # new_doc1 = []
        # new_doc2 = []
        # docs = doc.to_dict()
        # # print(doc.to_dict())
        # for i in filtered_med_indices.tolist():
        #     # print(i)
        #     new_doc1.append(docs[i[0]])
        # doc2 = CoNLL.dict2conll(new_doc1, 'texts_similar_filtered_med_conll')
        #
        # for i in filtered_threshold_indices.tolist():
        #     new_doc2.append(docs[i[0]])
        # doc3 = CoNLL.dict2conll(new_doc2, 'texts_similar_filtered_threshold_conll')
        # # with open('stanza/final_scores.csv', 'a') as f:
        # #     for i in config.scores[0]:
        # #             f.write(str(i[0]))
        # #             f.write('\n')
        return doc

    def __call__(self, doc):
        assert any([isinstance(doc, str), isinstance(doc, list),
                    isinstance(doc, Document)]), 'input should be either str, list or Document'
        doc = self.process(doc)
        return doc


