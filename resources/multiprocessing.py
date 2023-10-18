from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection, wait

from tqdm.autonotebook import tqdm

from typing import Callable, Tuple, Iterable, Union, Dict, Any

####################################################################################################
# Multiprocessing:                                                                                 #
####################################################################################################

class _queueWorker(Process):
    def __init__(self, target:Callable[...,None], connection:Connection, bar:Union[tqdm,None]=None, daemon:Union[bool,None]=None, **kwargs) -> None:
        super().__init__(daemon=daemon)
        self._target = target
        self._queue  = connection
        self._kwargs = kwargs

    def run(self) -> None:
        while True:
            # take next batch of arguments from the queue:
            try: args, kwargs = self._queue.recv()
            except EOFError:
                self._queue.close()
                return

            # add static variables to kwargs:
            kwargs.update(self._kwargs)

            # run calculation:
            result = self._target(*args, **kwargs)
            self._queue.send(result)

class ArgumentQueue:
    def __init__(self, target:Callable[...,Any], args:Union[Iterable[Iterable[Any]],None], kwargs:Union[Iterable[Dict[str, Any]],None]=None, bar:Union[tqdm,None]=None, **barargs) -> None:
        if args is None:
            self._size = len(kwargs)
            self._args = [[]] * self._size
            self._kwargs = kwargs

        elif kwargs is None:
            self._size = len(args)
            self._args = args
            self._kwargs = [{}] * self._size

        else:
            assert len(args) == len(kwargs)
            self._size = len(args)
            self._args = args
            self._kwargs = kwargs

        self._target = target
        self._index = 0

        if bar is None: self._bar = tqdm(total=self._size, **barargs)
        else:           self._bar = bar
        self._close_bar = (bar is None)

    def __repr__(self) -> str:
        output = ''
        for args, kwargs in zip(self._args, self._kwargs):
            entry = []

            for arg in args:
                entry.append(str(arg))
                             
            for kw in kwargs:
                entry.append(f'{kw}={str(kwargs[kw])}')

            output += f'({", ".join(entry)}),\n'
        return output[:-2]

    def __len__(self) -> int:
        return self._size

    def __call__(self, n_workers:int=5, **kwargs):
        # update progress bar:
        inc = self._bar.total / max(self._size, 1)
        self._bar.reset()

        # start workers:
        workers = []
        connections = []
        for _ in range(n_workers):
            # get next batch:
            try: args = self.pop()
            except StopIteration: break

            # create pipe and worker:
            conn, worker_conn = Pipe()
            worker = _queueWorker(self._target, worker_conn, **kwargs)

            # append to lists:
            workers.append(worker)
            connections.append(conn)

            # deliver first task:
            conn.send(args)
            worker.start()

        # wait for result:
        while len(connections) > 0:
            for conn in wait(connections):
                output = conn.recv()
                try: conn.send(self.pop())
                except StopIteration:
                    connections.remove(conn)
                    conn.close()
                    continue
                finally: 
                    self._bar.update(inc)
                    yield output

        # force update progress bar:
        self._bar.refresh()

        # join workers:
        for worker in workers:
            worker.join()

        # close bar:
        if self._close_bar:
            self._bar.close()

    def pop(self) -> Tuple[int, Iterable[Any],Dict[str, Any]]:
        # raise exception if empty:
        if self._index >= self._size:
            raise StopIteration()

        # get next batch of arguments:
        args = self._args[self._index]
        kwargs = self._kwargs[self._index]

        # increase counter
        self._index += 1

        return args, kwargs