import chainer.cuda

from chainermn.communicators import _base
from chainermn.communicators import _communication_utility
from chainermn.communicators import _memory_utility
from chainermn import nccl

import numpy as np


class PureNcclCommunicator(_base.CommunicatorBase):

    def __init__(self, mpi_comm, allreduce_grad_dtype=None):
        super(PureNcclCommunicator, self).__init__(mpi_comm, True)
        if nccl.get_version() < 2000:
            raise RuntimeError(
                'PureNcclCommunicator is only supported on NCCL 2.0+')
        self._init_ranks()

        # We have to delay the initialization of communicators. This is because
        # NCCL's communicators use the current CUDA devices at the time of
        # initialization. Therefore, we have to initialize NCCL communicators
        # after users set the devices to use.
        self.nccl_comm = None

        self.gpu_tmp_buffer = _memory_utility.DeviceMemory()
        self.gpu_allreduce_buffer_a = _memory_utility.DeviceMemory()
        self.gpu_allreduce_buffer_b = _memory_utility.DeviceMemory()

        if allreduce_grad_dtype is not None:
            self.allreduce_grad_dtype = np.dtype(allreduce_grad_dtype)
            if self.allreduce_grad_dtype.kind != 'f':
                raise ValueError(
                    'allreduce_grad_dtype must be'
                    'numpy.float16, numpy.float32,'
                    'numpy.float64, or None.')
        else:
            self.allreduce_grad_dtype = None

    def _init_ranks(self):
        my_ranks = _communication_utility.init_ranks(self.mpi_comm)
        assert my_ranks[0] == self.mpi_comm.rank
        self.intra_rank = my_ranks[1]
        self.intra_size = my_ranks[2]
        self.inter_rank = my_ranks[3]
        self.inter_size = my_ranks[4]

    def _init_comms(self):
        if self.nccl_comm is not None:
            return

        if not nccl._available:
            raise RuntimeError(
                'NCCL is not available. '
                'Please confirm that NCCL is enabled in CuPy.'
            )

        self.nccl_comm = _communication_utility.init_nccl_comm(self.mpi_comm)

    def broadcast_data(self, model):
        _communication_utility.broadcast_naive(self.mpi_comm, model)

    def allreduce_grad(self, model, stream=None):
        self._init_comms()
        if stream is None:
            stream = chainer.cuda.Stream.null

        params = _memory_utility.extract_params(model)
        grad_dtype = _get_param_grad_dtype(params[0])
        if self.allreduce_grad_dtype is None:
            allreduce_grad_dtype = grad_dtype
        else:
            allreduce_grad_dtype = self.allreduce_grad_dtype
        n_elems = sum(param.grad.size for param in params)
        self._assign(grad_dtype, allreduce_grad_dtype, n_elems)
        self._pack_params_to_buffer(params, grad_dtype, allreduce_grad_dtype,
                                    n_elems)
        if stream != chainer.cuda.Stream.null:
            chainer.cuda.Stream.null.synchronize()
        self.nccl_comm.allReduce(self.gpu_allreduce_buffer_a.ptr(),
                                 self.gpu_allreduce_buffer_b.ptr(), n_elems,
                                 _get_nccl_type_id(allreduce_grad_dtype),
                                 nccl.NCCL_SUM, stream.ptr)
        if stream != chainer.cuda.Stream.null:
            stream.synchronize()
        ret = self.gpu_allreduce_buffer_b.array(n_elems,
                                                dtype=allreduce_grad_dtype) \
            * (1.0/self.size)
        allreduce_grad_n_bytes = allreduce_grad_dtype.itemsize * n_elems
        self.gpu_allreduce_buffer_b.from_device(ret, allreduce_grad_n_bytes)
        self._unpack_params_from_buffer(params, grad_dtype,
                                        allreduce_grad_dtype, n_elems)

    def _assign(self, grad_dtype, allreduce_grad_dtype, n_elems):
        allreduce_grad_n_bytes = allreduce_grad_dtype.itemsize * n_elems
        self.gpu_allreduce_buffer_a.assign(allreduce_grad_n_bytes)
        self.gpu_allreduce_buffer_b.assign(allreduce_grad_n_bytes)
        if grad_dtype != allreduce_grad_dtype:
            grad_n_bytes = grad_dtype.itemsize * n_elems
            self.gpu_tmp_buffer.assign(grad_n_bytes)

    def _pack_params_to_buffer(self, params, grad_dtype, allreduce_grad_dtype,
                               n_elems):
        if grad_dtype == allreduce_grad_dtype:
            _memory_utility.pack_params(
                params, allreduce_grad_dtype.itemsize, 'grad',
                self.gpu_allreduce_buffer_a)
        else:
            _memory_utility.pack_params(
                params, grad_dtype.itemsize, 'grad',
                self.gpu_tmp_buffer)
            ret = self.gpu_tmp_buffer.array(n_elems,
                                            dtype=grad_dtype).astype(
                                                allreduce_grad_dtype)
            allreduce_grad_n_bytes = allreduce_grad_dtype.itemsize * n_elems
            self.gpu_allreduce_buffer_a.from_device(ret,
                                                    allreduce_grad_n_bytes)

    def _unpack_params_from_buffer(self, params, grad_dtype,
                                   allreduce_grad_dtype, n_elems):
        if grad_dtype == allreduce_grad_dtype:
            _memory_utility.unpack_params(
                params, allreduce_grad_dtype.itemsize, 'grad',
                self.gpu_allreduce_buffer_b)
        else:
            ret = self.gpu_allreduce_buffer_b.array(
                n_elems,
                dtype=allreduce_grad_dtype).astype(grad_dtype)
            grad_n_bytes = grad_dtype.itemsize * n_elems
            self.gpu_tmp_buffer.from_device(ret, grad_n_bytes)
            _memory_utility.unpack_params(
                params, grad_dtype.itemsize, 'grad', self.gpu_tmp_buffer)


def _get_param_grad_dtype(param):
    return param.grad.dtype


def _get_nccl_type_id(dtype):
    if dtype == np.float16:
        return nccl.NCCL_FLOAT16
    elif dtype == np.float32:
        return nccl.NCCL_FLOAT32
    elif dtype == np.float64:
        return nccl.NCCL_FLOAT64
    else:
        raise ValueError(
            'dtype must be float16, float32, or float64.')
