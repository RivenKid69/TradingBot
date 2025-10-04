# Имя файла: shared_memory_vec_env.py
import multiprocessing as mp
import numpy as np
import numpy as _np  # добавим alias на всякий случай
import time
import threading
import weakref
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, CloudpickleWrapper
from collections import OrderedDict
from gymnasium import spaces
from gymnasium.spaces.utils import flatten, flatten_space, unflatten
import atexit, signal
from typing import Any
try:
    from multiprocessing.context import BrokenBarrierError
except Exception:  # Python 3.12: no BrokenBarrierError in multiprocessing
    from threading import BrokenBarrierError

DTYPE_TO_CSTYLE = {
    np.float32: 'f',
    np.float64: 'd',
    np.bool_: 'b',
    np.int8: 'b',
    np.int32: 'i',
    np.int64: 'l',
    np.uint8: 'B',
    np.int16: 'h',
    'uint8': 'B',
    'int16': 'h',
}

def _safe_close_unlink(handle: Any) -> None:
    """Attempt to close/unlink shared memory handles without crashing."""

    if handle is None:
        return

    # Некоторые прокси объекты (например, ShareableList) держат ссылку на
    # реальный SharedMemory в атрибуте ``shm``.  Освободим и его тоже.
    inner = getattr(handle, "shm", None)
    if inner is not None and inner is not handle:
        _safe_close_unlink(inner)

    for attr in ("close", "unlink"):
        method = getattr(handle, attr, None)
        if method is None:
            continue
        try:
            method()
        except AttributeError:
            # multiprocessing.Array на некоторых платформах выбрасывает
            # AttributeError при закрытии — игнорируем.
            pass
        except FileNotFoundError:
            # unlink() мог быть вызван ранее — это нормально.
            pass

def worker(rank, num_envs, env_fn_wrapper, actions_shm, obs_shm, rewards_shm, dones_shm, info_queue, barrier, reset_signal, close_signal, obs_dtype, action_dtype, action_shape, obs_shape, action_is_structured, base_seed: int = 0):
    try:
        # 1. Создаем среду и получаем numpy-представления
        env = env_fn_wrapper.var()
        if not hasattr(env, "rank"):
            env.rank = rank

        # рассчитываем seed для данного воркера
        seed = int(base_seed) + int(rank)
        # инициализируем глобальный генератор numpy для совместимости
        np.random.seed(seed)
        # собственный генератор среды (если используется)
        env._rng = np.random.default_rng(seed)
        # НЕ трогаем env.observation_space до reset(); используем форму, переданную из родителя
        actions_np = np.frombuffer(actions_shm.get_obj(), dtype=action_dtype).reshape((num_envs,) + action_shape)
        obs_np = np.frombuffer(obs_shm.get_obj(), dtype=obs_dtype).reshape((num_envs,) + obs_shape)
        rewards_np = np.frombuffer(rewards_shm.get_obj(), dtype=np.float32)
        dones_np = np.frombuffer(dones_shm.get_obj(), dtype=np.bool_)

        # 2. Первоначальный reset с заданным seed
        try:
            obs, info = env.reset(seed=seed)
        except TypeError:
            if hasattr(env, "seed"):
                try:
                    env.seed(seed)
                except Exception:
                    pass
            obs, info = env.reset()
        obs_np[rank] = obs
        dones_np[rank] = False
        info_queue.put((rank, info))
        barrier.wait()

        # 3. Основной цикл работы
        while True:
            if close_signal.value:
                break    # graceful‑shutdown (PATCH‑ID:P12_P7_closecheck)
            barrier.wait()

            if close_signal.value:
                break # graceful-shutdown
            if reset_signal.value:
                # === ЛОГИКА СБРОСА ===
                obs, info = env.reset()
                obs_np[rank] = obs
                dones_np[rank] = False # Явно сбрасываем флаг завершения
                info_queue.put((rank, info))
            else:
                # === ЛОГИКА ШАГА (осталась прежней) ===
                if action_is_structured:
                    flat_action = np.array(actions_np[rank], copy=True)
                    action = unflatten(env.action_space, flat_action)
                else:
                    action = actions_np[rank]
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                if done:
                    obs, _ = env.reset() # Внутренний авто-сброс после завершения эпизода

                info_queue.put((rank, info))
                obs_np[rank] = obs
                rewards_np[rank] = reward
                dones_np[rank] = done

            barrier.wait()
        env.close()
        # --- cleanup shared-memory (PATCH-ID:P12_P7_shmcleanup) ---
        for _shm in (obs_shm, actions_shm, rewards_shm, dones_shm):
            _safe_close_unlink(_shm)
        return

    except BrokenBarrierError:
        # Барьер был сломан другим процессом (вероятно, из-за ошибки). Просто выходим.
        return
    except Exception as e:
        # В этом процессе произошла ошибка.
        print(f"!!! Worker {rank} crashed: {e}")
        # Сообщаем об ошибке главному процессу через очередь
        info_queue.put((rank, {"__error__": e, "traceback": str(e)}))
        # Ломаем барьер, чтобы другие процессы не зависли
        try:
            barrier.abort()
        except:
            pass
        # Завершаем процесс
        return



class SharedMemoryVecEnv(VecEnv):
    def __init__(self, env_fns, worker_timeout: float = 300.0, base_seed: int = 0):
        self.num_envs = len(env_fns)
        # Ждем, пока дочерние процессы не будут готовы
        self.waiting = False
        self.closed = False
        self._base_seed = base_seed
        
        # Создаем временную среду, чтобы получить размерности пространств
        temp_env = env_fns[0]()

        # ЛЕНИВЫЙ ИНИТ: если спейсы ещё не выставлены в __init__, дергаем reset()
        needs_reset = (
            getattr(temp_env, "action_space", None) is None or
            getattr(temp_env, "observation_space", None) is None or
            getattr(getattr(temp_env, "action_space", None), "dtype", None) is None
        )
        if needs_reset:
            try:
                temp_env.reset()
            except TypeError:
                # на случай сигнатуры reset(seed=None, options=None)
                temp_env.reset(seed=None)

        self.action_space = getattr(temp_env, "action_space", None)
        self.observation_space = getattr(temp_env, "observation_space", None)
        if self.action_space is None or self.observation_space is None:
            raise RuntimeError(
                "Env didn't expose action/observation spaces even after reset(). "
                "Set spaces in __init__ or ensure reset() defines them."
            )

        self._action_is_structured = isinstance(self.action_space, (spaces.Dict, spaces.Tuple))
        if self._action_is_structured:
            self._flat_action_space = flatten_space(self.action_space)
        else:
            self._flat_action_space = self.action_space

        # --- Robust dtype/shape discovery for action space -----------------
        # Некоторые среды (особенно с дискретными действиями) могут не
        # выставлять shape/dtype до первого reset() или вовсе держать их в
        # виде None.  Нам нужно гарантированно определить и dtype, и форму
        # буфера действий, иначе расчёт размера общей памяти упадёт.

        # 1) Попробуем взять dtype/shape напрямую
        action_dtype_raw = getattr(self._flat_action_space, "dtype", None)
        action_shape = getattr(self._flat_action_space, "shape", None)

        # 2) Если что-то не определено, используем sample() пространства
        action_sample_np = None
        if action_dtype_raw is None or action_shape is None or (
            isinstance(action_shape, tuple)
            and any(dim is None for dim in action_shape)
        ):
            # sample() может вернуть python-скаляры – приводим к numpy-массиву
            action_sample = self._flat_action_space.sample()
            action_sample_np = np.asarray(
                action_sample,
                dtype=action_dtype_raw if action_dtype_raw is not None else None,
            )
            if action_dtype_raw is None:
                action_dtype_raw = action_sample_np.dtype
            if action_shape is None or (
                isinstance(action_shape, tuple)
                and any(dim is None for dim in action_shape)
            ):
                action_shape = action_sample_np.shape

        # 3) Нормализуем dtype к numpy-классу (np.float32, np.int64, ...)
        if self._action_is_structured and action_dtype_raw is not None and _np.dtype(action_dtype_raw).type is np.float64:
            action_dtype_raw = np.float32
        act_type = _np.dtype(action_dtype_raw).type
        if act_type not in DTYPE_TO_CSTYLE:
            raise TypeError(
                f"Unsupported action dtype {action_dtype_raw} "
                f"(normalized: {act_type}). Known: {list(DTYPE_TO_CSTYLE.keys())}"
            )
        action_type_code = DTYPE_TO_CSTYLE[act_type]

        self._flat_action_dtype = act_type

        if hasattr(temp_env, "close"):
            temp_env.close()

        obs_shape = self.observation_space.shape
        obs_dtype = self.observation_space.dtype

        # 4) Для скаляров (shape == ()) np.prod даст 1 и reshape корректно
        if action_shape is None:
            # fallback: если пространство экзотическое и даже sample() не дал
            # shape, трактуем действие как скаляр.
            action_shape = ()
        elif action_sample_np is not None and action_sample_np.shape != action_shape:
            # Подстрахуемся: приведём форму к кортежу из sample(), если она
            # отличается (бывает с python-скалярами, где space.shape -> (),
            # а sample_np.shape -> () тоже; условие просто не сработает).
            action_shape = action_sample_np.shape
        action_shape = tuple(action_shape)  # гарантируем кортеж
        self._flat_action_shape = action_shape

        # 1. Создаем массивы в общей памяти с помощью multiprocessing.Array
        # 'f' - float, 'd' - double, 'b' - boolean
        try:
            obs_type_code = DTYPE_TO_CSTYLE[obs_dtype.type]
        except KeyError as e:
            raise KeyError(
                f"Unsupported dtype {e} found in observation or action space. "
                f"Please add it to the DTYPE_TO_CSTYLE dictionary in shared_memory_vec_env.py"
            )

        self.obs_shm = mp.Array(obs_type_code, self.num_envs * int(np.prod(obs_shape)))
        self.actions_shm = mp.Array(action_type_code, self.num_envs * int(np.prod(self._flat_action_shape)))
        self.rewards_shm = mp.Array('f', self.num_envs) # Награды почти всегда float32
        self.dones_shm = mp.Array('B', self.num_envs) # 'B' = unsigned char, более безопасный тип для bool

        # 2. Создаем numpy-представления для удобной работы в главном процессе
        self.obs_np = np.frombuffer(self.obs_shm.get_obj(), dtype=obs_dtype).reshape((self.num_envs,) + obs_shape)
        self.actions_np = np.frombuffer(self.actions_shm.get_obj(), dtype=self._flat_action_dtype).reshape((self.num_envs,) + self._flat_action_shape)
        self.rewards_np = np.frombuffer(self.rewards_shm.get_obj(), dtype=np.float32)
        self.dones_np = np.frombuffer(self.dones_shm.get_obj(), dtype=np.bool_)
        
        # 3. Создаем барьер для синхронизации. 
        #    Количество участников = количество работников + 1 (главный процесс)
        self.info_queue = mp.Queue()
        self.barrier = mp.Barrier(self.num_envs + 1)
        self.reset_signal = mp.Value('b', False)
        self.close_signal = mp.Value('b', False)
        
        # 4. Запускаем дочерние процессы
        self.processes = []
        for i, env_fn in enumerate(env_fns):
            process = mp.Process(
                target=worker,
                args=(
                    i,
                    self.num_envs,
                    CloudpickleWrapper(env_fn),
                    self.actions_shm,
                    self.obs_shm,
                    self.rewards_shm,
                    self.dones_shm,
                    self.info_queue,
                    self.barrier,
                    self.reset_signal,
                    self.close_signal,
                    obs_dtype,
                    self._flat_action_dtype,
                    self._flat_action_shape,
                    obs_shape,
                    self._action_is_structured,
                    self._base_seed,
                )
            )
            process.daemon = True
            process.start()
            self.processes.append(process)

        # Ждем, пока все работники выполнят первоначальный reset
        self.barrier.wait()
        # После этого барьера self.obs_np уже содержит корректные начальные наблюдения

        # Сохраняем info, полученную во время первичного reset, чтобы сделать
        # её доступной через стандартное поле VecEnv.reset_infos.
        initial_infos = [{} for _ in range(self.num_envs)]
        for _ in range(self.num_envs):
            rank, info = self.info_queue.get()
            initial_infos[rank] = info
        self.reset_infos = initial_infos
        
        # --- leak-guard: регистрируем все shm-сегменты и аварийное закрытие ---
        self._shm_arrays = [self.obs_shm, self.actions_shm, self.rewards_shm, self.dones_shm]

        # atexit: на всякий случай закроем и удалим сегменты при завершении процесса
        atexit.register(lambda wr=weakref.ref(self): getattr(wr(), "close", lambda: None)())

        # сигналы: корректно чистим сегменты при SIGINT/SIGTERM и чейним старый хэндлер
        for _sig in (signal.SIGINT, signal.SIGTERM):
            try:
                _prev = signal.getsignal(_sig)
                def _handler(signum, frame, _prev=_prev):
                    try:
                        if not getattr(self, "closed", False):
                            self.close()
                    finally:
                        if callable(_prev):
                            _prev(signum, frame)
                signal.signal(_sig, _handler)
            except Exception:
                pass
        self.worker_timeout = worker_timeout
        self._last_step_t0 = 0.0
        self._wd_stop = threading.Event()
        self._wd = threading.Thread(
            target=self._watchdog_loop, args=(weakref.ref(self),), daemon=True
        )
        self._wd.start()

        super().__init__(self.num_envs, self.observation_space, self.action_space)

    def _flatten_single_action(self, action):
        if self._action_is_structured:
            flat = flatten(self.action_space, action)
            flat = np.asarray(flat, dtype=self._flat_action_dtype)
            return flat.reshape(self._flat_action_shape)
        arr = np.asarray(action, dtype=self._flat_action_dtype)
        return arr.reshape(self._flat_action_shape)

    def _flatten_action_batch(self, actions):
        if self._action_is_structured:
            if isinstance(actions, np.ndarray):
                if actions.shape == (self.num_envs,) + self._flat_action_shape:
                    if actions.dtype != self._flat_action_dtype:
                        actions = actions.astype(self._flat_action_dtype, copy=False)
                    return np.ascontiguousarray(actions)
                if self.num_envs == 1 and actions.shape == self._flat_action_shape:
                    reshaped = actions.reshape((self.num_envs,) + self._flat_action_shape)
                    if reshaped.dtype != self._flat_action_dtype:
                        reshaped = reshaped.astype(self._flat_action_dtype, copy=False)
                    return np.ascontiguousarray(reshaped)
            if isinstance(actions, dict):
                actions_iter = [actions]
            else:
                actions_iter = list(actions)
            if len(actions_iter) != self.num_envs:
                raise ValueError(
                    f"Expected {self.num_envs} structured actions, got {len(actions_iter)}"
                )
            flat_actions = [self._flatten_single_action(act) for act in actions_iter]
            stacked = np.stack(flat_actions, axis=0)
            if stacked.dtype != self._flat_action_dtype:
                stacked = stacked.astype(self._flat_action_dtype, copy=False)
            return np.ascontiguousarray(stacked)

        arr = np.asarray(actions, dtype=self._flat_action_dtype)
        expected_shape = (self.num_envs,) + self._flat_action_shape
        if arr.shape != expected_shape:
            if arr.size != int(np.prod(expected_shape)):
                raise ValueError(
                    f"Cannot reshape actions of shape {arr.shape} into {expected_shape}"
                )
            arr = arr.reshape(expected_shape)
        return np.ascontiguousarray(arr)

    def step_async(self, actions):
        # Копируем действия в общую память
        flattened_actions = self._flatten_action_batch(actions)
        self.actions_np[...] = flattened_actions
        self.waiting = True
        self._last_step_t0 = time.perf_counter()     # ← отметка старта шага
        # Сигнализируем работникам, что можно начинать шаг (снимаем барьер)
        self.barrier.wait()

    def step_wait(self):
        if not self.waiting:
            raise RuntimeError("Trying to wait for a step that was not requested")

        try:
            # Добавляем таймаут к ожиданию
            self.barrier.wait(timeout=self.worker_timeout)
        except BrokenBarrierError:
            self._force_kill()
            self.close()
            # Уточняем возможное сообщение об ошибке
            raise RuntimeError("A worker process timed out, crashed, or the barrier was aborted.")

        infos = [{} for _ in range(self.num_envs)]
        for _ in range(self.num_envs):
            rank, info = self.info_queue.get()

            if "__error__" in info:
                self.close()
                # Воссоздаем ошибку в главном процессе
                raise RuntimeError(f"Worker {rank} crashed: {info['traceback']}")

            infos[rank] = info

        self.waiting = False
        # Возвращаем копии данных, чтобы исключить передачу указателей на
        # разделяемую память вызывающему коду.
        return self.obs_np.copy(), self.rewards_np.copy(), self.dones_np.copy(), infos
    def _force_kill(self):
        """Жёсткое завершение воркеров + попытка освободить ресурсы."""
        try:
            for p in self.processes:
                if p.is_alive():
                    p.terminate()
            for p in self.processes:
                p.join(timeout=1.0)
        except Exception:
            pass

    def _watchdog_loop(self, self_ref):
        """Сторожок: если шаг завис дольше 2× worker_timeout — глушим воркеров."""
        # Период опроса — 0.25 сек; порог — 2× worker_timeout
        poll = 0.25
        while not self._wd_stop.is_set():
            time.sleep(poll)
            self_obj = self_ref()
            if self_obj is None:
                break
            # если идёт шаг и тикает таймер — проверим, не зависли ли
            if self.waiting and self._last_step_t0:
                elapsed = time.perf_counter() - self._last_step_t0
                if elapsed > max(0.0, float(self.worker_timeout)) * 2.0:
                    # фиксируем зависание: ломаем барьер и жёстко гасим воркеров
                    try:
                        self.barrier.abort()
                    except Exception:
                        pass
                    self._force_kill()
                    # дальнейшая логика: даём close() привести всё в порядок
                    # и выходим из сторожка
                    break

    def reset(self):
        while not self.info_queue.empty():
            self.info_queue.get_nowait()
        # 1. Подаем сигнал на сброс
        self.reset_signal.value = True

        try:
            # 2. Отпускаем воркеров, чтобы они НАЧАЛИ сброс
            self.barrier.wait(timeout=self.worker_timeout)

            # 4. Ждем, пока воркеры ЗАВЕРШАТ сброс
            self.barrier.wait(timeout=self.worker_timeout)

        except BrokenBarrierError:
            self._force_kill()
            self.close()
            raise RuntimeError("A worker process timed out or crashed during reset.")
        finally:
            self.reset_signal.value = False

        # 5. Собираем инфо-сообщения от воркеров
        infos = [{} for _ in range(self.num_envs)]
        for _ in range(self.num_envs):
            rank, info = self.info_queue.get()
            infos[rank] = info

        # 6. Обновляем reset_infos и возвращаем только наблюдения (совместимо со
        #    стандартным интерфейсом VecEnv). Информацию можно получить из
        #    self.reset_infos сразу после вызова reset().
        self.reset_infos = infos

        # Возвращаем копию наблюдений, чтобы исключить передачу указателей на
        # разделяемый буфер.
        return self.obs_np.copy()

    def close(self):
        """
        Graceful-shutdown: сообщаем воркерам, ждём их и полностью
        освобождаем shared_memory-сегменты.
        """
        if getattr(self, "closed", False):
            return

        # 1) сигнал воркерам: пора выходить
        self.close_signal.value = True

        try:
            # 2) пробуем совместно выйти через барьер
            self.barrier.wait(timeout=self.worker_timeout)
        except BrokenBarrierError:
            pass  # барьер уже сломан — игнорируем

        # 3) даём воркерам время корректно завершиться
        for p in self.processes:
            p.join(timeout=1.0)

        # 4) если кто-то всё ещё жив — убиваем
        for p in self.processes:
            if p.is_alive():
                p.terminate()
                p.join()

        # 5) закрываем очередь info
        self.info_queue.close()
        self.info_queue.join_thread()

        # 6) освобождаем и безопасно unlink-уем все shm-сегменты
        for _arr in (getattr(self, "_shm_arrays", []) or []):
            _safe_close_unlink(_arr)

        # останавливаем watchdog
        try:
            self._wd_stop.set()
            if hasattr(self, "_wd") and self._wd is not None:
                self._wd.join(timeout=0.5)
        except Exception:
            pass

        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.close()
        finally:
            return False  # не подавлять исключения

    def __del__(self):
        try:
            if not getattr(self, "closed", True):
                self.close()
        except Exception:
            pass

    # === Stub implementations required by VecEnv base class ===
    def _indices(self, indices):
        if indices is None:
            return range(self.num_envs)
        if isinstance(indices, int):
            indices = [indices]
        return indices

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._indices(indices)]

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [None for _ in self._indices(indices)]

    def get_attr(self, attr_name, indices=None):
        return [None for _ in self._indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        pass
