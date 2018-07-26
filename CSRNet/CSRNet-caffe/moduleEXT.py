# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

'''
    ModuleEXT is the extended version of mxnet.module.Module
    It's based on the code of incubator-mxnet(https://github.com/apache/incubator-mxnet)
'''

import mxnet as mx
import numpy as np
import logging
import warnings
from mxnet.model import _create_kvstore, _initialize_kvstore, _update_params, _update_params_on_kvstore

class ModuleEXT(mx.module.Module): 
    def __init__(self, *args, **kwargs):
        super(ModuleEXT, self).__init__(*args, **kwargs)
        self.set_l2norm_grad_clip(used = False)

    def set_preload_optimizer_states(self, fname = None, prefix = None, epoch = None):
        if fname is not None:
            self._preload_opt_states = fname
        else:
            self._preload_opt_states = '%s-%04d.states' % (prefix, epoch)

    def set_l2norm_grad_clip(self, clip_gradients = 35, clip_gradients_global = True, verbose = False, used = True):
        self.use_l2norm_grad_clip = used 
        self.grad_clip_verbose = verbose
        self.clip_gradients = clip_gradients
        self.clip_gradients_global = clip_gradients_global

    def init_optimizer(self, kvstore='local', optimizer='sgd',
                       optimizer_params=(('learning_rate', 0.01),), force_init=False):
        """Installs and initializes optimizers.

        Parameters
        ----------
        kvstore : str or KVStore
            Default `'local'`.
        optimizer : str or Optimizer
            Default `'sgd'`
        optimizer_params : dict
            Default `(('learning_rate', 0.01),)`. The default value is not a dictionary,
            just to avoid pylint warning of dangerous default values.
        force_init : bool
            Default ``False``, indicating whether we should force re-initializing the
            optimizer in the case an optimizer is already installed.
        """
        assert self.binded and self.params_initialized

        if self.optimizer_initialized and not force_init:
            self.logger.warning('optimizer already initialized, ignoring...')
            return

        if self._params_dirty:
            self._sync_params_from_devices()

        (kvstore, update_on_kvstore) = \
                mx.model._create_kvstore(kvstore, len(self._context), self._arg_params)

        batch_size = self._exec_group.batch_size
        if kvstore and 'dist' in kvstore.type and '_sync' in kvstore.type:
            batch_size *= kvstore.num_workers
        rescale_grad = 1.0 / batch_size


        idx2name = {}
        if update_on_kvstore:
            idx2name.update(enumerate(self._exec_group.param_names))
        else:
            for k in range(len(self._context)):
                idx2name.update({i*len(self._context)+k: n
                                 for i, n in enumerate(self._exec_group.param_names)})
        name2idx = {}
        for k, v in idx2name.items():
            if v not in name2idx:
                name2idx[v] = []
            name2idx[v].append(k)

        if isinstance(optimizer, str):
            optimizer_params = dict(optimizer_params)
            if 'rescale_grad' not in optimizer_params:
                optimizer_params['rescale_grad'] = rescale_grad
            optimizer = mx.optimizer.create(optimizer,
                                   sym=self.symbol, param_idx2name=idx2name,
                                   **optimizer_params)
        else:
            assert isinstance(optimizer, mx.optimizer.Optimizer)
            if optimizer.rescale_grad != rescale_grad:
                #pylint: disable=no-member
                warnings.warn(
                    "Optimizer created manually outside Module but rescale_grad " +
                    "is not normalized to 1.0/batch_size/num_workers (%s vs. %s). "%(
                        optimizer.rescale_grad, rescale_grad) +
                    "Is this intended?", stacklevel=2)
            if len(optimizer.idx2name):
                warnings.warn("The idx2name of the optimizer is overwrote by ModuleEXT")
            # overwrite optimizer.idx2name
            optimizer.idx2name = idx2name.copy()

        self._param_idx2name = idx2name 
        self._param_name2idx = name2idx
        self._optimizer = optimizer
        self._kvstore = kvstore
        self._update_on_kvstore = update_on_kvstore
        self._updater = None

        if kvstore:
            if self._compression_params:
                kvstore.set_gradient_compression(self._compression_params)
            # copy initialized local parameters to kvstore
            _initialize_kvstore(kvstore=kvstore,
                                param_arrays=self._exec_group.param_arrays,
                                arg_params=self._arg_params,
                                param_names=self._param_names,
                                update_on_kvstore=update_on_kvstore)
        if update_on_kvstore:
            kvstore.set_optimizer(self._optimizer)
        else:
            self._updater = mx.optimizer.get_updater(optimizer)

        self.optimizer_initialized = True

        if self._preload_opt_states is not None:
            self.load_optimizer_states(self._preload_opt_states)
            self._preload_opt_states = None

    def forward_backward(self, data_batch):
        """A convenient function that calls both ``forward`` and ``backward``."""
        self.forward(data_batch, is_train=True)
        self.backward()
        if self.use_l2norm_grad_clip:
            # 2-Norm Grad Clip
            self.l2norm_grad_clip()

    def l2norm_grad_clip(self):
        if self.clip_gradients_global:
            # Global
            sum_grad = 0.0
            num_execs = len(self._exec_group.execs)
            for grads in self._exec_group.grad_arrays:
                for ctx_id in range(num_execs):
                    grad = grads[ctx_id]
                    if grad is not None:
                        sum_grad += mx.nd.square(mx.nd.norm(grad)).asscalar() # ||W||_2
            l2norm_grad = np.sqrt(sum_grad / num_execs) / self._exec_group.batch_size

            if l2norm_grad > self.clip_gradients:
                scale_factor = self.clip_gradients / l2norm_grad
                if self.grad_clip_verbose:
                    logging.info("Gradient clipping: scaling down gradients (L2 norm %f > %f) by scale factor %f" % (l2norm_grad, self.clip_gradients, scale_factor))
                for grads in self._exec_group.grad_arrays:
                    for ctx_id in range(num_execs):
                        grad = grads[ctx_id]
                        if grad is not None:
                            grads[ctx_id][:] *= scale_factor
        else:
            # each device independently
            for ctx_id in range(len(self._exec_group.execs)):
                sum_grad = 0.0
                for grads in self._exec_group.grad_arrays:
                    grad = grads[ctx_id]
                    if grad is None:
                        continue
                    sum_grad += mx.nd.square(mx.nd.norm(grad)).asscalar() # ||W||_2
                l2norm_grad = np.sqrt(sum_grad) / self._exec_group.batch_size
                if l2norm_grad > self.clip_gradients:
                    scale_factor = self.clip_gradients / l2norm_grad
                    for grads in self._exec_group.grad_arrays:
                        grad = grads[ctx_id]
                        if grad is not None:
                            grad[:] *= scale_factor
                    if self.grad_clip_verbose and ctx_id == 0:
                        logging.info("Gradient clipping: scaling down gradients (L2 norm %f > %f) by scale factor %f" % (l2norm_grad, self.clip_gradients, scale_factor))

    def print_gradients(self, names = None):
        # names: list or None
        assert names is None or type(names) == list, "the type of names must be list or None"

        # only print the information of device 0
        grad_arrays = self._exec_group.grad_arrays

        if names is not None:
            print_index = self.get_idxes_from_names(names)
        else:
            print_index = range(len(grad_arrays))

        for index in print_index:
            grads = grad_arrays[index]
            name = self._param_idx2name[index]
            grad = grads[0] # device 0
            assert grad is not None, "The gradient %s doesn't exist" % name
            mean = mx.nd.mean(grad).asscalar()
            std  = mx.nd.norm(grad - mean).asscalar() / np.sqrt(grad.size)
            logging.info("%s: [mean: %e, var: %e]" % (name, mean, std))
        logging.info("--------------------------")

    def get_idxes_from_names(self, names):
        # names: list
        assert type(names) == list, "the type of names must be list"
        idx = []
        for name in names:
            assert name in self._param_name2idx, "%s is not any parameter name" % name
            idx.extend(self._param_name2idx[name])
        return idx
