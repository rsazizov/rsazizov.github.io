---
layout: post
title: PyTorch and Shared Memory
---

**DRAFT**

Recently I stumbled upon a weird error when working on some code in PyTorch. Here is the minimal 
reproducible example:

```python
import torch as th
from torch.utils.data import DataLoader, Dataset


class RandomDataset(Dataset):

    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index) -> dict:
        return {
            'label1': False,
            'label2': th.tensor([1]),
            'label3': th.tensor([0.1, 0.2, 0.3, 0.4]),
            'label4': th.tensor([1.0]),
            'label5': index
        }


def main() -> None:
    dataset = RandomDataset(20000)
    data_loader = DataLoader(dataset, num_workers=1)

    batches = []

    for batch in data_loader:
        batches.append(batch)

    print('Done')


if __name__ == '__main__':
    main()
```

Here we have a `RandomDataset` class which simply repeat the same dict object `size` times. In `main` we create
a data loader with a single worker and iterate over batches appending them to `batches` list.

If you try to run this, you would get a long stack trace that ends with this error message:

```
    storage = cls._new_shared_fd_cpu(fd, size)
RuntimeError: unable to mmap 4 bytes from file <filename not specified>: Cannot allocate memory (12)
```

We are told that PyTorch can't allocate memory. I checked `htop` and there was plenty of RAM available. So
what's the issue here?

## DataLoader Multiprocessing

When we created a data loader, we specified `num_workers = 1`. PyTorch allows several worker processes to 
process data in parallel. The reasoning behind this is to avoid data loading bottlenecks, as they limit
GPU utilization. A typical PyTorch training loop would look something like this:

```
while data_loader is not empty:
    fetch a batch from data_loader
    model(batch)
    optimizer step
```

This is essentially a sequential process. First, we ask our DataLoader to sample the dataset and collate a
batch for us to use. Depending on your preprocessing stages (e.g. loading from a slow storage, 
augmentations, etc...), this step can take a long time that GPU will spend idling. Since data can usually be
loaded and preprocessed in parallel, a natural idea is to utilize some sort of parallelism inside the data
loader. This is exactly what PyTorch does under the hood.

![](/assets/img/torch-shmem/workers.png)

PyTorch uses a simple work-queue pattern. DataLoader spawns several worker processes and creates
an index queue for each worker. When you iterate over the data loader, it splits dataset indices
among the workers and distributes them to index queues. Workers then do the actual sampling (`next(dataset)`)
and send the result to the common data queue, which is then read back by the data loader.

Let's talk a bit more about these queues. Due to [GIL](), Python can't have proper threading
parallelism, so PyTorch has to resolve to processes. Using processes avoids GIL, though
introduces some other complications like a need for IPC. 

Linux (and other operating systems, but I'll focus on Linux here) has memory protection mechanisms which were
put in place to prevent processes from reading and mutating each other's memory. This is somewhat of a necessity due
to security reasons, though it is inconvenient when we actually want two processes to share data (like in this case).
The way around this is to use IPC (Inter-Process Communication) mechanisms. Data and Index queues in the diagram
above are implemented using modified Python's multithreading.Queue object, which is essentially a unix pipe. They are 
easy to work with and thread-safe, which is convenient. However, the problem with this IPC is that it copies data around
a lot. 

Suppose a worker process samples the dataset which allocates a new tensor. Tensor is allocated inside the worker process'
memory and to transfer it to the main process you would need to copy the entire tensor, which is slow and inefficient.
For sharing tensors, PyTorch uses a special kind of IPC called shared memory. Shared memory are essentially memory regions
that can be accessed by several processes at the same time, which means no expensive memory copying is needed.

## Shared Tensors

So when a Tensor is put inside the data queue, PyTorch creates a new shared memory regions and places tensor's data
in the region. This happens inside `reduce_storage` function in
[`torch/multiprocessing/reductions.py`](https://github.com/pytorch/pytorch/blob/main/torch/multiprocessing/reductions.py#L428).
`reduce_storage` is called when a tensor is pushed into a queue (see [`queue.py`](https://github.com/pytorch/pytorch/blob/main/torch/multiprocessing/queue.py#L16C11-L16C11)). 

```python
def reduce_storage(storage):
    ...
        fd, size = storage._share_fd_cpu_()
        df = multiprocessing.reduction.DupFd(fd)
        cache_key = fd_id(fd)
        metadata = (df, size)
        rebuild = rebuild_storage_fd  # type: ignore[assignment]
    ...
```

`storage._share_fd_cpu_()` is what actually places the tensor data into shared memory. It is bound to 
[`THPStorage_shareFd`](https://github.com/pytorch/pytorch/blob/11602ac564c0e3178b38a65e09be13644322d303/torch/csrc/StorageSharing.cpp#L194)
function which uses [`MapAllocator`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/MapAllocator.cpp#L262C35-L262C35) 
class with `ALLOCATOR_MAPPED_SHARED` which instructs `MapAllocator` to create a new shared memory region.

Thanks to PyTorch, we can actually access this API from Python. `Tensor` class implements two related methods:
[`memory_share_()`](https://pytorch.org/docs/stable/generated/torch.Tensor.share_memory_.html)
(which is equivalent to `_share_fd_cpu_` for cpu tensors) and 
[`is_shared()`](https://pytorch.org/docs/stable/generated/torch.Tensor.is_shared.html).

Let's play around with them a little. On Linux, shared memory can be inspected using `lsof /dev/shm` command.
PyTorch uses `torch/` prefix with its tensors. Let's try them:

```python
import torch as th

x = th.tensor([1,2,3])
x.data_ptr()
Out[4]: 94350617509440

x.is_shared()
Out[5]: False

x.share_memory_()
Out[6]: tensor([1, 2, 3])

x.is_shared()
Out[7]: True

x.data_ptr()
Out[8]: 140618151415808
```

We first create a tensor `x`. `data_ptr()` allows us to see the address where the actual data is stored. 
We call `share_memory_()` method and `data_ptr()` returns a new address. Running `lsof /dev/shm | grep torch`:

```
python3.1 2149720 pc3966  DEL    REG   0,26          170061957 /dev/shm/torch_2149720_1191275662_0
```

We can see that python process created a new shmem region named `torch_2149720_1191275662_0` which is our actual
tensor.

## Shared Memory Limits

That was a lot of information to handle, though now we can safely return to the initial code snippet and identify
the problem:

```python
batches = []

for batch in data_loader:
    print(batch.is_shared()) # True
    ...
```

So tensors arrive shared by default. This is a problem here because we *save* batches in `batches` list,
which means that `batch` tensor is not garbage collected when it goes out of scope and the shared memory region 
remains alive. 

