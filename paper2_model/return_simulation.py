"""Slot-based store-and-forward simulation matching paper section 2.2."""
from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from paper1.paper2_dataset import Paper2Dataset, _read_events
from paper2_model.channel import ChannelConfig, ChannelModel, evaluate_schedule
from paper2_model.capacity_diagnostic import base_station_capacity_upper_bound


@dataclass(frozen=True)
class DataUnit:
    data_id: int
    source_uav_id: int
    original_bits: float
    collection_slot_zero_based: int
    collection_complete_time_s: float | None = None

    @property
    def collection_slot_paper_one_based(self):
        return self.collection_slot_zero_based + 1


@dataclass(frozen=True)
class SimulationConfig:
    buffer_capacity_bits: float = 1e12
    absolute_tolerance_bits: float = 1e-6
    relative_tolerance: float = 1e-10

    def validated(self):
        if not np.isfinite(self.buffer_capacity_bits) or self.buffer_capacity_bits <= 0:
            raise ValueError('buffer_capacity_bits must be positive and finite')
        if self.absolute_tolerance_bits < 0 or self.relative_tolerance < 0:
            raise ValueError('tolerances must be nonnegative')
        return self


@dataclass(frozen=True)
class SlotState:
    slot_index: int
    queues_bits: np.ndarray
    node_total_buffer_bits: np.ndarray


def _integer_value(value, field):
    """Return an integer-valued scalar without silently truncating identifiers."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{field} must be a finite integer, got {value!r}')
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        return int(value)
    raise ValueError(f'{field} must be a finite integer, got {value!r}')


def load_data_units(dataset, data_per_collection_bytes=None):
    """Load one immutable data unit per existing collection event without rewriting input files."""
    events = _read_events(Path(dataset.folder) / 'collection_events.csv')
    units = []
    for event in events:
        bits = (int(data_per_collection_bytes) * 8 if data_per_collection_bytes is not None
                else int(event['data_bits']))
        if bits <= 0:
            raise ValueError('every data unit must contain a positive number of bits')
        units.append(DataUnit(int(event['data_id']), int(event['source_uav_id']), float(bits),
                              int(event['enqueue_slot_zero_based']),
                              float(event['collection_complete_time_s'])))
    units.sort(key=lambda unit: unit.data_id)
    if [unit.data_id for unit in units] != list(range(len(units))):
        raise ValueError('data_id must be contiguous zero-based for queue matrix rows')
    return units


class ReturnSimulation:
    """Synchronous queue engine; schedules are decisions, all state changes occur at slot end."""
    def __init__(self, dataset, channel_model, data_units, config=None):
        self.dataset = dataset
        self.channel_model = channel_model
        self.channel_config = channel_model.config
        self.config = (config or SimulationConfig()).validated()
        self.units = list(data_units)
        self.num_uavs = _integer_value(dataset.scenario['num_uavs'], 'num_uavs')
        self.slot_count = _integer_value(dataset.scenario['slot_count'], 'slot_count')
        try:
            self.slot_seconds = float(dataset.scenario['slot_seconds'])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'slot_seconds must be positive and finite, got '
                             f'{dataset.scenario["slot_seconds"]!r}') from exc
        if self.num_uavs <= 0:
            raise ValueError(f'num_uavs must be positive, got {self.num_uavs!r}')
        if self.slot_count <= 0:
            raise ValueError(f'slot_count must be a positive integer, got {self.slot_count!r}')
        if not np.isfinite(self.slot_seconds) or self.slot_seconds <= 0:
            raise ValueError(f'slot_seconds must be positive and finite, got {self.slot_seconds!r}')
        if self.channel_model.num_uavs != self.num_uavs:
            raise ValueError('dataset and channel model UAV counts differ')
        data_ids, sources, collection_slots, exact_times, original_bits = [], [], [], [], []
        for index, unit in enumerate(self.units):
            data_id = _integer_value(unit.data_id, f'data_units[{index}].data_id')
            source = _integer_value(unit.source_uav_id, f'data_units[{index}].source_uav_id')
            slot = _integer_value(unit.collection_slot_zero_based,
                                  f'data_units[{index}].collection_slot_zero_based')
            if isinstance(unit.original_bits, (bool, np.bool_)):
                raise ValueError(f'data_units[{index}].original_bits must be positive and finite, '
                                 f'got {unit.original_bits!r}')
            try:
                bits = float(unit.original_bits)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f'data_units[{index}].original_bits must be positive and finite, '
                                 f'got {unit.original_bits!r}') from exc
            if not np.isfinite(bits) or bits <= 0:
                raise ValueError(f'data_units[{index}].original_bits must be positive and finite, '
                                 f'got {unit.original_bits!r}')
            if not 0 <= source < self.num_uavs:
                raise ValueError(f'data_units[{index}].source_uav_id out of range: {unit.source_uav_id!r}')
            if not 0 <= slot < self.slot_count:
                raise ValueError(f'data_units[{index}].collection_slot_zero_based out of range: '
                                 f'{unit.collection_slot_zero_based!r}')
            if unit.collection_complete_time_s is None:
                exact_time = None
            else:
                if isinstance(unit.collection_complete_time_s, (bool, np.bool_)):
                    raise ValueError(f'data_units[{index}].collection_complete_time_s must be finite '
                                     f'and nonnegative, got {unit.collection_complete_time_s!r}')
                try:
                    exact_time = float(unit.collection_complete_time_s)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(f'data_units[{index}].collection_complete_time_s must be finite '
                                     f'and nonnegative, got {unit.collection_complete_time_s!r}') from exc
                if not np.isfinite(exact_time) or exact_time < 0:
                    raise ValueError(f'data_units[{index}].collection_complete_time_s must be finite '
                                     f'and nonnegative, got {unit.collection_complete_time_s!r}')
                actual_slot = int(np.floor(exact_time / self.slot_seconds))
                if actual_slot != slot:
                    raise ValueError(f'data_units[{index}].collection_complete_time_s '
                        f'{unit.collection_complete_time_s!r} belongs to slot {actual_slot}, not {slot}; '
                        'slot k is [k*dt,(k+1)*dt) and the right boundary belongs to the next slot')
            data_ids.append(data_id); sources.append(source); collection_slots.append(slot)
            exact_times.append(exact_time); original_bits.append(bits)
        if data_ids != list(range(len(self.units))):
            raise ValueError('data_id values must be unique, ordered, and contiguous 0...K-1; '
                             f'got {data_ids!r}')
        takeoff = dataset.node_states.get('takeoff_time_s') if hasattr(dataset, 'node_states') else None
        returns = dataset.node_states.get('return_time_s') if hasattr(dataset, 'node_states') else None
        if takeoff is not None and returns is not None:
            takeoff, returns = np.asarray(takeoff, float), np.asarray(returns, float)
            if takeoff.shape != (self.num_uavs,) or returns.shape != (self.num_uavs,):
                raise ValueError('takeoff_time_s and return_time_s must have one value per UAV')
            for index, (source, exact_time) in enumerate(zip(sources, exact_times)):
                if exact_time is not None and not (takeoff[source] <= exact_time < returns[source]):
                    raise ValueError(f'data_units[{index}].collection_complete_time_s {exact_time!r} '
                        f'is outside source UAV {source} online interval '
                        f'[{takeoff[source]!r}, {returns[source]!r})')
        self.original_bits = np.asarray(original_bits, float)
        self.tolerances = (self.config.absolute_tolerance_bits +
                           self.config.relative_tolerance * self.original_bits)
        self.collection_slots = np.asarray(collection_slots, int)
        self.sources = np.asarray(sources, int)
        self.collection_times_s = np.array([
            ((slot + 1) * self.slot_seconds if exact_time is None else exact_time)
            for slot, exact_time in zip(collection_slots, exact_times)], float)

    def tolerance(self, data_id):
        return self.tolerances[data_id]

    def run(self, schedule_provider, allocation_provider=None, baseline_name='user supplied schedule'):
        unit_count = len(self.units)
        queues = np.zeros((self.num_uavs, unit_count), float)
        availability = np.full((self.num_uavs, unit_count), np.inf)
        base_received = np.zeros(unit_count, float)
        delivery_slots = np.full(unit_count, -1, int)
        generated = np.zeros(unit_count, bool)
        peak_buffers = np.zeros(self.num_uavs)
        buffer_history = np.zeros((self.slot_count, self.num_uavs))
        first_tx_slots = np.full(unit_count, -1, int)
        positive_tx_slot_counts = np.zeros(unit_count, int)
        scheduled_history = np.zeros((self.slot_count, self.num_uavs), bool)
        sent_history = np.zeros((self.slot_count, self.num_uavs))
        online_backlog_history = np.zeros((self.slot_count, self.num_uavs), bool)
        direct_rate_history = np.full((self.slot_count, self.num_uavs), np.nan)
        generated_cumulative = 0.
        buffer_violations, exit_records, slot_records = [], {}, []
        return_times = np.asarray(self.dataset.node_states.get(
            'return_time_s', np.full(self.num_uavs, self.slot_count * self.slot_seconds)), float)

        for slot in range(self.slot_count):
            snapshot = self.channel_model.snapshot(slot)
            queue_start = queues.copy()
            queue_start.setflags(write=False)
            totals = queue_start.sum(axis=1)
            totals.setflags(write=False)
            state = SlotState(slot, queue_start, totals)
            eligible = snapshot.scheduling_eligible[:self.num_uavs]
            online_backlog_history[slot] = eligible & (totals > self.config.absolute_tolerance_bits)
            online = np.flatnonzero(eligible)
            if len(online):
                b = self.channel_config.subchannel_bandwidth_hz
                noise = self.channel_config.noise_psd_w_per_hz * b
                direct_rate_history[slot, online] = b * np.log2(
                    1 + self.channel_config.transmit_power_w *
                    snapshot.power_gain[online, self.channel_model.base_station_id] / noise)
            schedule = self._get_schedule(schedule_provider, slot, state, snapshot)
            for tx, _, _ in schedule:
                if tx < self.num_uavs:
                    scheduled_history[slot, tx] = True
            evaluation = evaluate_schedule(snapshot, schedule)
            allocations = (self._fifo_allocations(state, availability, base_received, evaluation)
                           if allocation_provider is None else
                           allocation_provider(slot, state, snapshot, evaluation))
            normalized = self._validate_allocations(queue_start, base_received, evaluation, allocations)

            transmitted = 0.
            positive_ids = set()
            for tx, rx, data_id, amount in normalized:
                queues[tx, data_id] -= amount
                if abs(queues[tx, data_id]) <= self.tolerance(data_id):
                    queues[tx, data_id] = 0.
                if rx == self.channel_model.base_station_id:
                    base_received[data_id] += amount
                else:
                    was_empty = queues[rx, data_id] <= self.tolerance(data_id)
                    queues[rx, data_id] += amount
                    if was_empty and amount > self.tolerance(data_id):
                        availability[rx, data_id] = slot + 1
                transmitted += amount
                if amount > self.tolerance(data_id):
                    sent_history[slot, tx] += amount
                    positive_ids.add(data_id)
                    if first_tx_slots[data_id] < 0:
                        first_tx_slots[data_id] = slot
            if positive_ids:
                positive_tx_slot_counts[list(positive_ids)] += 1

            arrivals = np.flatnonzero(self.collection_slots == slot)
            for data_id in arrivals:
                source = self.sources[data_id]
                queues[source, data_id] += self.original_bits[data_id]
                availability[source, data_id] = slot + 1
                generated[data_id] = True
            generated_cumulative += float(self.original_bits[arrivals].sum())

            for data_id in np.flatnonzero((delivery_slots < 0) & generated):
                tol = self.tolerance(data_id)
                if base_received[data_id] >= self.original_bits[data_id] - tol:
                    if base_received[data_id] > self.original_bits[data_id] + tol:
                        raise RuntimeError(f'base received more than original data for unit {data_id}')
                    base_received[data_id] = self.original_bits[data_id]
                    delivery_slots[data_id] = slot + 1

            occupancy = queues.sum(axis=1)
            buffer_history[slot] = occupancy
            peak_buffers = np.maximum(peak_buffers, occupancy)
            for uav in np.flatnonzero(occupancy > self.config.buffer_capacity_bits + self.config.absolute_tolerance_bits):
                buffer_violations.append(dict(slot_zero_based=slot, slot_paper_one_based=slot + 1,
                    uav_id=int(uav), occupancy_bits=float(occupancy[uav]),
                    capacity_bits=float(self.config.buffer_capacity_bits)))
            self._check_conservation(slot, queues, base_received, generated)

            slot_end = (slot + 1) * self.slot_seconds
            for uav in range(self.num_uavs):
                if uav not in exit_records and return_times[uav] <= slot_end + 1e-12:
                    ids = np.flatnonzero(queues[uav] > self.tolerances)
                    exit_records[uav] = dict(uav_id=uav, communication_exit_time_s=float(return_times[uav]),
                        exit_slot_paper_one_based=slot + 1, stranded_bits=float(queues[uav].sum()),
                        stranded_data_unit_ids=ids.tolist())
            slot_records.append(dict(slot_zero_based=slot, slot_paper_one_based=slot + 1,
                scheduled_link_count=len(schedule), transmitted_bits=float(transmitted),
                generated_bits=float(self.original_bits[arrivals].sum()),
                generated_cumulative_bits=generated_cumulative,
                base_received_cumulative_bits=float(base_received.sum()),
                system_undelivered_bits=float(generated_cumulative - base_received.sum()),
                total_uav_buffer_bits=float(occupancy.sum())))

        return self._result(baseline_name, queues, base_received, delivery_slots,
                            peak_buffers, buffer_history, buffer_violations, exit_records, slot_records,
                            first_tx_slots, positive_tx_slot_counts, scheduled_history, sent_history,
                            online_backlog_history, direct_rate_history)

    @staticmethod
    def _get_schedule(provider, slot, state, snapshot):
        if callable(provider):
            return list(provider(slot, state, snapshot))
        return list(provider.get(slot, []))

    def _fifo_allocations(self, state, availability, base_received, evaluation):
        allocations, sent = [], np.zeros_like(state.queues_bits)
        base_room = self.original_bits - base_received
        for link in evaluation['links']:
            tx, rx = link['transmitter'], link['receiver']
            capacity = link['rate_bps'] * self.slot_seconds
            candidates = np.flatnonzero(state.queues_bits[tx] > self.tolerances)
            candidates = sorted(candidates, key=lambda k: (availability[tx, k], self.units[k].data_id))
            for data_id in candidates:
                held = state.queues_bits[tx, data_id] - sent[tx, data_id]
                amount = min(capacity, held)
                if rx == self.channel_model.base_station_id:
                    amount = min(amount, base_room[data_id])
                if amount > self.tolerance(data_id):
                    allocations.append((tx, rx, int(data_id), float(amount)))
                    sent[tx, data_id] += amount
                    if rx == self.channel_model.base_station_id:
                        base_room[data_id] -= amount
                    capacity -= amount
                if capacity <= self.config.absolute_tolerance_bits:
                    break
        return allocations

    def _validate_allocations(self, queue_start, base_received, evaluation, allocations):
        links = {(link['transmitter'], link['receiver']):
                 link['rate_bps'] * self.slot_seconds for link in evaluation['links']}
        used_capacity = {link: 0. for link in links}
        sent = np.zeros_like(queue_start)
        delivered = np.zeros(len(self.units))
        normalized = []
        for allocation in allocations:
            if len(allocation) != 4:
                raise ValueError('allocation must be (tx, rx, data_id, bits)')
            tx, rx, data_id, amount = allocation
            tx = _integer_value(tx, 'allocation.tx')
            rx = _integer_value(rx, 'allocation.rx')
            data_id = _integer_value(data_id, 'allocation.data_id')
            if not 0 <= tx <= self.channel_model.base_station_id:
                raise ValueError(f'allocation.tx out of range: {tx!r}')
            if not 0 <= rx <= self.channel_model.base_station_id:
                raise ValueError(f'allocation.rx out of range: {rx!r}')
            if not 0 <= data_id < len(self.units):
                raise ValueError(f'allocation.data_id out of range: {data_id!r}')
            if (tx, rx) not in links:
                raise ValueError(f'allocation link ({tx}, {rx}) was not legally scheduled')
            try:
                amount = float(amount)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f'allocation.bits must be nonnegative and finite, got {amount!r}') from exc
            if not np.isfinite(amount) or amount < 0:
                raise ValueError(f'allocation.bits must be nonnegative and finite, got {amount!r}')
            used_capacity[(tx, rx)] += amount
            sent[tx, data_id] += amount
            if rx == self.channel_model.base_station_id:
                delivered[data_id] += amount
            normalized.append((tx, rx, data_id, amount))
        for link, amount in used_capacity.items():
            if amount > links[link] + self.config.absolute_tolerance_bits:
                raise ValueError(f'link {link} allocation exceeds slot capacity')
        for tx, data_id in zip(*np.nonzero(sent)):
            if sent[tx, data_id] > queue_start[tx, data_id] + self.tolerance(data_id):
                raise ValueError(f'UAV {tx} sends unavailable bits of data unit {data_id}')
        for data_id in np.flatnonzero(delivered):
            if delivered[data_id] > self.original_bits[data_id] - base_received[data_id] + self.tolerance(data_id):
                raise ValueError(f'base delivery exceeds remaining size of data unit {data_id}')
        return normalized

    def _check_conservation(self, slot, queues, base_received, generated):
        actual = queues.sum(axis=0) + base_received
        expected = np.where(generated, self.original_bits, 0.)
        bad = np.flatnonzero(np.abs(actual - expected) > self.tolerances)
        if len(bad):
            raise RuntimeError(f'data conservation failed in slot {slot} for units {bad[:10].tolist()}')

    def _result(self, name, queues, base_received, delivery_slots, peak_buffers, buffer_history,
                buffer_violations, exit_records, slot_records, first_tx_slots,
                positive_tx_slot_counts, scheduled_history, sent_history,
                online_backlog_history, direct_rate_history):
        completed = delivery_slots >= 0
        delays = np.full(len(self.units), np.nan)
        delivery_times = np.full(len(self.units), np.nan)
        delivery_times[completed] = delivery_slots[completed] * self.slot_seconds
        delays[completed] = delivery_times[completed] - self.collection_times_s[completed]
        paper_slot_delays = np.full(len(self.units), np.nan)
        paper_slot_delays[completed] = ((delivery_slots[completed] - self.collection_slots[completed] - 1)
                                        * self.slot_seconds)
        started = first_tx_slots >= 0
        first_tx_times = np.full(len(self.units), np.nan)
        first_tx_times[started] = first_tx_slots[started] * self.slot_seconds
        pre_send_wait = np.full(len(self.units), np.nan)
        pre_send_wait[started] = first_tx_times[started] - self.collection_times_s[started]
        post_start_to_complete = np.full(len(self.units), np.nan)
        finished_started = completed & started
        post_start_to_complete[finished_started] = (
            delivery_times[finished_started] - first_tx_times[finished_started])
        decomposition_error = (np.max(np.abs(delays[finished_started] -
            pre_send_wait[finished_started] - post_start_to_complete[finished_started]))
            if np.any(finished_started) else 0.)
        total = float(self.original_bits.sum())
        residual_locations = []
        for uav, data_id in zip(*np.nonzero(queues > self.tolerances[None, :])):
            residual_locations.append(dict(uav_id=int(uav), data_id=int(data_id),
                                           remaining_bits=float(queues[uav, data_id])))
        all_complete = bool(completed.all())
        completed_delays = delays[completed]
        completed_pre_wait = pre_send_wait[completed]
        completed_post_wait = post_start_to_complete[completed]
        undelivered = total - float(base_received.sum())
        conservation_error = abs(total - float(base_received.sum()) - float(queues.sum()))
        partial_incomplete = (~completed) & (base_received > self.tolerances)
        max_peak = float(peak_buffers.max(initial=0.))
        return dict(baseline_name=name, data_units=self.units, base_received_bits=base_received,
            channel_config=asdict(self.channel_config), random_seed=int(self.channel_config.random_seed),
            simulation_config=asdict(self.config), slot_count=self.slot_count,
            slot_seconds=self.slot_seconds, simulation_horizon_s=self.slot_count * self.slot_seconds,
            trajectory_simulation_end_time_s=float(self.dataset.scenario.get(
                'simulation_end_time_s', self.slot_count * self.slot_seconds)),
            delivery_slot_paper_one_based=delivery_slots, delivery_time_s=delivery_times,
            delays_s=delays, paper_slot_delays_s=paper_slot_delays,
            first_tx_slot_zero_based=first_tx_slots, first_tx_time_s=first_tx_times,
            pre_first_send_wait_s=pre_send_wait,
            post_first_send_to_complete_s=post_start_to_complete,
            positive_tx_slot_count=positive_tx_slot_counts,
            delay_decomposition_max_error_s=float(decomposition_error),
            complete=completed, complete_data_unit_rate=float(completed.mean()),
            data_unit_count=len(self.units), complete_data_unit_count=int(completed.sum()),
            delivered_bit_ratio=float(base_received.sum() / total), total_generated_bits=total,
            total_generated_bytes=total / 8, delivered_bits=float(base_received.sum()),
            delivered_bytes=float(base_received.sum()) / 8,
            undelivered_data_bits=undelivered, undelivered_data_bytes=undelivered / 8,
            all_data_complete=all_complete,
            all_data_mean_delay_s=float(np.mean(delays)) if all_complete else None,
            completed_data_conditional_mean_delay_s=float(completed_delays.mean()) if completed.any() else None,
            completed_data_p95_delay_s=float(np.percentile(completed_delays, 95)) if completed.any() else None,
            completed_data_max_delay_s=float(completed_delays.max()) if completed.any() else None,
            completed_pre_first_send_mean_s=float(completed_pre_wait.mean()) if completed.any() else None,
            completed_pre_first_send_p95_s=float(np.percentile(completed_pre_wait, 95)) if completed.any() else None,
            completed_post_first_send_mean_s=float(completed_post_wait.mean()) if completed.any() else None,
            completed_post_first_send_p95_s=float(np.percentile(completed_post_wait, 95)) if completed.any() else None,
            incomplete_data_unit_count=int((~completed).sum()),
            partially_received_incomplete_data_unit_count=int(partial_incomplete.sum()),
            residual_data_bits=float(queues.sum()), residual_locations=residual_locations,
            final_queues_bits=queues, peak_buffer_bits_per_uav=peak_buffers,
            slot_buffer_bits_per_uav=buffer_history,
            slot_uav_scheduled=scheduled_history, slot_uav_sent_bits=sent_history,
            slot_uav_online_backlog=online_backlog_history,
            slot_uav_direct_rate_bps=direct_rate_history,
            total_backlog_peak_bits=float(buffer_history.sum(axis=1).max(initial=0.)),
            max_single_uav_peak_buffer_bits=max_peak,
            buffer_capacity_bits=float(self.config.buffer_capacity_bits),
            max_buffer_utilization=float(max_peak / self.config.buffer_capacity_bits),
            buffer_constraint_violated=bool(buffer_violations), buffer_violations=buffer_violations,
            performance_metrics_valid=not bool(buffer_violations),
            exit_stranded=[exit_records[u] for u in sorted(exit_records)],
            slot_records=slot_records, conservation_passed=bool(conservation_error <=
                self.config.absolute_tolerance_bits + self.config.relative_tolerance * total),
            final_conservation_error_bits=conservation_error)

    def direct_rate_greedy_schedule(self, slot, state, snapshot):
        """Direct-return rate-greedy baseline: best source-to-base rates, distinct subchannels."""
        nonempty = state.node_total_buffer_bits > self.config.absolute_tolerance_bits
        eligible = snapshot.scheduling_eligible[:self.num_uavs] & nonempty
        uavs = np.flatnonzero(eligible)
        if not len(uavs):
            return []
        b = self.channel_config.subchannel_bandwidth_hz
        noise = self.channel_config.noise_psd_w_per_hz * b
        rates = b * np.log2(1 + self.channel_config.transmit_power_w *
                            snapshot.power_gain[uavs, self.channel_model.base_station_id] / noise)
        count = min(self.channel_config.num_subchannels, len(uavs))
        selected = uavs[np.argsort(rates)[-count:][::-1]]
        return [(int(uav), self.channel_model.base_station_id, channel)
                for channel, uav in enumerate(selected)]

    def run_direct_rate_greedy(self):
        return self.run(self.direct_rate_greedy_schedule,
                        baseline_name='direct return: instantaneous-rate greedy')

    def make_round_robin_schedule(self):
        """Stable UAV-ID cyclic service; the pointer advances past the last selected UAV."""
        pointer = 0
        def schedule(slot, state, snapshot):
            nonlocal pointer
            eligible = (snapshot.scheduling_eligible[:self.num_uavs] &
                        (state.node_total_buffer_bits > self.config.absolute_tolerance_bits))
            selected = []
            for offset in range(self.num_uavs):
                uav = (pointer + offset) % self.num_uavs
                if eligible[uav]:
                    selected.append(uav)
                    if len(selected) == self.channel_config.num_subchannels:
                        break
            if selected:
                pointer = (selected[-1] + 1) % self.num_uavs
            return [(int(uav), self.channel_model.base_station_id, channel)
                    for channel, uav in enumerate(selected)]
        return schedule

    def run_round_robin(self):
        return self.run(self.make_round_robin_schedule(),
                        baseline_name='direct return: UAV-ID round robin')

    def oldest_data_first_schedule(self, slot, state, snapshot):
        """Select UAVs whose FIFO head has the greatest age; ties use UAV ID."""
        eligible = (snapshot.scheduling_eligible[:self.num_uavs] &
                    (state.node_total_buffer_bits > self.config.absolute_tolerance_bits))
        ranked = []
        slot_start = slot * self.slot_seconds
        for uav in np.flatnonzero(eligible):
            ids = np.flatnonzero(state.queues_bits[uav] > self.tolerances)
            head = min(ids, key=lambda data_id: (self.collection_times_s[data_id],
                                                 self.units[data_id].data_id))
            age = slot_start - self.collection_times_s[head]
            ranked.append((-age, int(uav)))
        selected = [uav for _, uav in sorted(ranked)[:self.channel_config.num_subchannels]]
        return [(uav, self.channel_model.base_station_id, channel)
                for channel, uav in enumerate(selected)]

    def run_oldest_data_first(self):
        return self.run(self.oldest_data_first_schedule,
                        baseline_name='direct return: oldest FIFO-head first')


def write_simulation_result(result, output_dir):
    """Write compact result tables; input files are never touched."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    units = result['data_units']
    summary = {key: value for key, value in result.items() if key not in {
        'data_units', 'base_received_bits', 'delivery_slot_paper_one_based', 'delivery_time_s',
        'delays_s', 'paper_slot_delays_s', 'first_tx_slot_zero_based', 'first_tx_time_s',
        'pre_first_send_wait_s', 'post_first_send_to_complete_s', 'positive_tx_slot_count',
        'complete', 'final_queues_bits', 'residual_locations', 'exit_stranded', 'slot_records',
        'slot_uav_scheduled', 'slot_uav_sent_bits', 'slot_uav_online_backlog',
        'slot_uav_direct_rate_bps'}}
    summary.pop('slot_buffer_bits_per_uav', None)
    summary['residual_location_count'] = len(result['residual_locations'])
    summary['exit_stranded'] = result['exit_stranded']
    summary['peak_buffer_bits_per_uav'] = result['peak_buffer_bits_per_uav'].tolist()
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    experiment_config = dict(channel_config=result['channel_config'],
        simulation_config=result['simulation_config'], random_seed=result['random_seed'],
        slot_count=result['slot_count'], slot_seconds=result['slot_seconds'],
        simulation_horizon_s=result['simulation_horizon_s'],
        trajectory_simulation_end_time_s=result['trajectory_simulation_end_time_s'],
        data_unit_count=len(units), data_per_collection_bytes=result.get('experiment_data_per_collection_bytes'),
        data_per_collection_bits=(result.get('experiment_data_per_collection_bytes') or 0) * 8,
        total_generated_bits=result['total_generated_bits'])
    for key in ('experiment_name', 'input_folder', 'output_folder', 'input_generation_config',
                'parameter_sources', 'input_validation', 'data_size_distribution',
                'data_size_seed', 'source_input_folder'):
        if key in result:
            experiment_config[key] = result[key]
    (output / 'experiment_config.json').write_text(
        json.dumps(experiment_config, ensure_ascii=False, indent=2), encoding='utf-8')
    with (output / 'data_units.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['data_id', 'source_uav_id', 'collection_complete_time_s',
                         'collection_slot_paper_one_based', 'original_bytes', 'original_bits',
                         'base_received_bytes', 'base_received_bits', 'remaining_bytes', 'remaining_bits',
                         'complete', 'first_tx_slot_zero_based', 'first_tx_time_s',
                         'pre_first_send_wait_s', 'positive_tx_slot_count',
                         'delivery_slot_paper_one_based', 'delivery_time_s',
                         'post_first_send_to_complete_s', 'delay_s'])
        for k, unit in enumerate(units):
            received = result['base_received_bits'][k]
            remaining = unit.original_bits - received
            writer.writerow([unit.data_id, unit.source_uav_id,
                unit.collection_complete_time_s, unit.collection_slot_paper_one_based,
                unit.original_bits / 8, unit.original_bits, received / 8, received,
                remaining / 8, remaining, bool(result['complete'][k]),
                int(result['first_tx_slot_zero_based'][k]) if result['first_tx_slot_zero_based'][k] >= 0 else '',
                result['first_tx_time_s'][k] if result['first_tx_slot_zero_based'][k] >= 0 else '',
                result['pre_first_send_wait_s'][k] if result['first_tx_slot_zero_based'][k] >= 0 else '',
                int(result['positive_tx_slot_count'][k]),
                int(result['delivery_slot_paper_one_based'][k]) if result['complete'][k] else '',
                result['delivery_time_s'][k] if result['complete'][k] else '',
                result['post_first_send_to_complete_s'][k] if result['complete'][k] else '',
                result['delays_s'][k] if result['complete'][k] else ''])
    with (output / 'uav_buffers.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['uav_id', 'generated_bits', 'base_received_source_bits',
                         'complete_source_units', 'source_unit_count', 'complete_source_unit_rate',
                         'completed_mean_delay_s', 'completed_p95_delay_s',
                         'undelivered_source_bits', 'peak_buffer_bits', 'final_buffer_bits',
                         'online_backlog_slots', 'service_opportunity_slots',
                         'longest_online_backlog_without_service_slots',
                         'longest_online_backlog_without_service_s', 'mean_available_direct_rate_bps',
                         'communication_exit_time_s', 'exit_slot_paper_one_based',
                         'exit_stranded_bits', 'exit_stranded_data_unit_ids'])
        exits = {row['uav_id']: row for row in result['exit_stranded']}
        for uav, peak in enumerate(result['peak_buffer_bits_per_uav']):
            record = exits.get(uav, {})
            ids = [k for k, unit in enumerate(units) if unit.source_uav_id == uav]
            generated = sum(units[k].original_bits for k in ids)
            received = float(result['base_received_bits'][ids].sum()) if ids else 0.
            completed = int(result['complete'][ids].sum()) if ids else 0
            completed_ids = [k for k in ids if result['complete'][k]]
            source_delays = result['delays_s'][completed_ids]
            online_backlog = result['slot_uav_online_backlog'][:, uav]
            served = result['slot_uav_sent_bits'][:, uav] > result['simulation_config']['absolute_tolerance_bits']
            longest = run = 0
            for missed in online_backlog & ~served:
                run = run + 1 if missed else 0
                longest = max(longest, run)
            rates = result['slot_uav_direct_rate_bps'][:, uav]
            writer.writerow([uav, generated, received, completed, len(ids),
                             completed / len(ids) if ids else 0.,
                             float(source_delays.mean()) if len(source_delays) else '',
                             float(np.percentile(source_delays, 95)) if len(source_delays) else '',
                             generated - received, peak, result['final_queues_bits'][uav].sum(),
                             int(online_backlog.sum()), int((online_backlog & served).sum()),
                             longest, longest * result['slot_seconds'],
                             float(np.nanmean(rates[online_backlog])) if online_backlog.any() else '',
                             record.get('communication_exit_time_s', ''),
                             record.get('exit_slot_paper_one_based', ''),
                             record.get('stranded_bits', ''),
                             ' '.join(map(str, record.get('stranded_data_unit_ids', [])))])
    with (output / 'slot_uav_metrics.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['slot_zero_based', 'slot_paper_one_based', 'slot_start_s', 'uav_id',
                         'buffer_end_bits', 'online_with_backlog_at_slot_start', 'scheduled',
                         'actual_sent_bits', 'available_direct_rate_bps'])
        for slot in range(result['slot_count']):
            for uav in range(result['slot_buffer_bits_per_uav'].shape[1]):
                rate = result['slot_uav_direct_rate_bps'][slot, uav]
                writer.writerow([slot, slot + 1, slot * result['slot_seconds'], uav,
                    result['slot_buffer_bits_per_uav'][slot, uav],
                    bool(result['slot_uav_online_backlog'][slot, uav]),
                    bool(result['slot_uav_scheduled'][slot, uav]),
                    result['slot_uav_sent_bits'][slot, uav], rate if np.isfinite(rate) else ''])
    with (output / 'residual_locations.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=['uav_id', 'data_id', 'remaining_bits'])
        writer.writeheader(); writer.writerows(result['residual_locations'])
    with (output / 'slot_metrics.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(result['slot_records'][0]))
        writer.writeheader(); writer.writerows(result['slot_records'])
    completed_ids = np.flatnonzero(result['complete'])
    top_ids = completed_ids[np.argsort(result['delays_s'][completed_ids])[::-1][:50]]
    with (output / 'top50_completed_delays.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['rank', 'data_id', 'source_uav_id', 'collection_complete_time_s',
                         'first_tx_time_s', 'delivery_time_s', 'pre_first_send_wait_s',
                         'post_first_send_to_complete_s', 'total_delay_s'])
        for rank, k in enumerate(top_ids, 1):
            writer.writerow([rank, units[k].data_id, units[k].source_uav_id,
                units[k].collection_complete_time_s, result['first_tx_time_s'][k],
                result['delivery_time_s'][k], result['pre_first_send_wait_s'][k],
                result['post_first_send_to_complete_s'][k], result['delays_s'][k]])
    os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / '.runtime' / 'matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    times = np.array([row['slot_paper_one_based'] for row in result['slot_records']], float) * result['slot_seconds']
    generated = np.array([row['generated_cumulative_bits'] for row in result['slot_records']])
    received = np.array([row['base_received_cumulative_bits'] for row in result['slot_records']])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(times, generated, label='Cumulative generated')
    ax.plot(times, received, label='Cumulative received at base station')
    ax.set(xlabel='Time (s)', ylabel='Data (bit)')
    ax.grid(True, alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(output / 'cumulative_generated_received.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(times, generated - received)
    ax.set(xlabel='Time (s)', ylabel='Undelivered data (bit)')
    ax.grid(True, alpha=.3); fig.tight_layout()
    fig.savefig(output / 'system_undelivered.png', dpi=160); plt.close(fig)

    source_counts = np.zeros(len(result['peak_buffer_bits_per_uav']), int)
    complete_counts = np.zeros_like(source_counts)
    for k, unit in enumerate(units):
        source_counts[unit.source_uav_id] += 1
        complete_counts[unit.source_uav_id] += int(result['complete'][k])
    rates = np.divide(complete_counts, source_counts, out=np.zeros_like(source_counts, float),
                      where=source_counts > 0)
    uavs = np.arange(len(source_counts))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(uavs, rates * 100)
    ax.set(xlabel='UAV ID', ylabel='Complete delivery rate (%)', xticks=uavs, ylim=(0, 105))
    ax.grid(True, axis='y', alpha=.3); fig.tight_layout()
    fig.savefig(output / 'uav_complete_delivery_rate.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(uavs, result['peak_buffer_bits_per_uav'] / 8e9)
    ax.axhline(result['buffer_capacity_bits'] / 8e9, color='red', linestyle='--',
               label=f'Buffer limit = {result["buffer_capacity_bits"] / 8e9:g} GB')
    ax.set(xlabel='UAV ID', ylabel='Peak buffer (decimal GB)', xticks=uavs)
    ax.grid(True, axis='y', alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(output / 'uav_peak_buffer.png', dpi=160); plt.close(fig)


def run_yard_baseline(input_folder, output_folder, channel_config=None, simulation_config=None):
    dataset = Paper2Dataset(input_folder)
    channel = ChannelModel(dataset, channel_config or ChannelConfig())
    units = load_data_units(dataset)
    capacity = base_station_capacity_upper_bound(dataset, model=channel,
        total_data_bits=sum(unit.original_bits for unit in units), data_unit_count=len(units))
    simulation = ReturnSimulation(dataset, channel, units,
                                  simulation_config or SimulationConfig(buffer_capacity_bits=1e12))
    result = simulation.run_direct_rate_greedy()
    fixed_amount = dataset.scenario.get('data_per_collection_bytes')
    result['experiment_data_per_collection_bytes'] = (int(fixed_amount)
                                                       if fixed_amount is not None else None)
    result['capacity_diagnostic'] = {key: value for key, value in capacity.items()
                                     if not isinstance(value, np.ndarray)}
    write_simulation_result(result, output_folder)
    return result
