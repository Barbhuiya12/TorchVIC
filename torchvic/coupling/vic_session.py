"""DaySession: VIC's day exchange (`exchange: day`) run as one simulate
call over a block of days, handed out a day at a time.

A one-day window is one basin.simulate call per day, and on a GPU every
call builds its set-up again and captures its CUDA graphs again (an eager
step and a capture per graph): on the ISC domain that made VIC 4-5 times
slower per day than in its 31-day windows. A session runs the call over a
block of days (62 by default) in a worker thread and stops it at every
day boundary in its day_hook (the mechanism the day-coupled modules use,
coupling/runtime.DayCoupler): the hook hands the day's outputs over --
simulate(day_outputs=True) forms them as a call over that day alone
returns them -- and waits for the next day's additions (precip_add,
soil_add), which the main thread works out while the worker waits. So the
runtime still runs, checks and commits VIC one day at a time, with every
other module in between, and VIC's graphs are captured once a block.

Nothing of the model changes: a run cut into windows is the single run
(tests/test_state_carry.py), and a day of the session is, bit for bit, the
one-day call from the state the day starts on (tests/test_vic_session.py).
The session ends -- and the next day starts a new one from the committed
state -- whenever the day asked for is not the next one of the session
from the state it handed out (a rejected or redone step, a checkpoint
loaded) and at the block's end.

Only the main thread or the worker runs at any time (each waits on the
other's queue), so the two share the device's stream as one thread would.
"""
import queue
import threading

import torch


class _Stop(Exception):
    """Raised in the worker's day hook to end the call early."""


class DaySession(object):
    """simulate over steps [s0, s1) of `driver` from `state`, a day at a
    time. `first()` returns the first day's outputs, `next(ret)` the next
    day's after handing the day hook `ret` (None, or (new soil moisture in
    the loop's layout, the column total added) as basin's day_hook
    returns it); `view` is the day hook's _DayView between two days (the
    next day's soil moisture and precipitation forcing).

    `ahead` > 0 (a VIC that nothing feeds and that runs alone, so no day
    of the block can be given anything and no other module's code runs
    beside it): after the first day the worker does not wait at the day
    boundaries; it computes up to `ahead` days beyond those handed out,
    while the main thread checks and commits the earlier ones. The days
    are the same (the hook returns nothing either way); the block's CUDA
    graphs are captured in its first day, when the main thread waits."""

    def __init__(self, driver, s0, s1, state, precip_add=None, ahead=0,
                 **sim):
        self.driver = driver
        self.s0, self.s1 = int(s0), int(s1)
        self.spd = int(driver.steps_per_day)
        #: the first step of the day `next` hands out
        self.next_step = self.s0
        #: the state of the last day handed out (the connector commits it)
        self.last_state = None
        self.view = None
        self.graphs = False
        self.alive = True
        self.ahead = int(ahead or 0)
        self._room = threading.Semaphore(self.ahead) if self.ahead else None
        self._stop = False
        self._q_out = queue.Queue()
        self._q_in = queue.Queue()
        n = self.s1 - self.s0
        infer = torch.is_inference_mode_enabled()
        grad = torch.is_grad_enabled()
        dev = getattr(driver, 'device', None)
        first_hook = [True]

        def hook(t_, view):
            out = view.day_outputs()
            if self._room is not None and not first_hook[0]:
                self._q_out.put(('day', out, None, bool(view.graphs)))
                self._room.acquire()
                if self._stop:
                    raise _Stop()
                return None
            first_hook[0] = False
            self._q_out.put(('day', out, view, bool(view.graphs)))
            msg = self._q_in.get()
            if msg[0] == 'stop':
                raise _Stop()
            return msg[1]

        def work():
            try:
                if dev is not None and torch.device(dev).type == 'cuda' \
                        and torch.device(dev).index is not None:
                    torch.cuda.set_device(dev)
                with torch.inference_mode(infer), \
                        torch.set_grad_enabled(grad and not infer):
                    out = driver.run(self.s0, self.s1, state,
                                     precip_add=precip_add, day_hook=hook,
                                     day_outputs=True,
                                     tail_from=n - self.spd, **sim)
                self._q_out.put(('last', out, None, None))
            except _Stop:
                self._q_out.put(('stopped', None, None, None))
            except BaseException as e:                   # noqa: BLE001
                self._q_out.put(('error', e, None, None))

        self._thread = threading.Thread(target=work, daemon=True,
                                        name='torchvic-vic-session')
        self._thread.start()

    @property
    def open(self):
        """Another day of the block can be handed out."""
        return self.alive and (self.view is not None or (
            self._room is not None and self.next_step > self.s0))

    def _take(self):
        kind, val, view, graphs = self._q_out.get()
        if kind == 'error':
            self.alive = False
            self._thread.join()
            raise val
        if kind not in ('day', 'last'):
            self.alive = False
            raise RuntimeError('vic session: the call stopped')
        self.view = view
        if graphs is not None:
            self.graphs = graphs
        self.next_step += self.spd
        if kind == 'last':
            # the block's last day: the call has returned
            self.alive = False
            self._thread.join()
        return val

    def first(self):
        return self._take()

    def next(self, ret=None):
        if not self.open:
            raise RuntimeError('vic session: no day left in the call')
        if self.view is None:
            # running ahead: nothing to hand the hook
            if ret is not None:
                raise RuntimeError('vic session: a day ahead takes nothing')
            self._room.release()
            return self._take()
        self.view = None
        self._q_in.put(('go', ret))
        return self._take()

    def close(self):
        """End the call (the worker leaves simulate) and wait for it."""
        if self.alive and self._thread.is_alive():
            self._stop = True
            if self._room is not None:
                self._room.release()
            self._q_in.put(('stop', None))
            while True:
                kind = self._q_out.get()[0]
                if kind in ('stopped', 'error', 'last'):
                    break
                # a day handed out after the stop: decline it
                if self._room is not None:
                    self._room.release()
                self._q_in.put(('stop', None))
            self._thread.join()
        self.alive = False
        self.view = None
