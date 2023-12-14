# https://github.com/briansemrau/MIDI-LLM-tokenizer

# MIT License

# Copyright (c) 2023 Brian Semrau

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import json
import random
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, floor, log
from typing import Dict, Iterator, List, Optional, Tuple

import mido


@dataclass
class VocabConfig:
    # Number of note events. Should be 128.
    note_events: int
    # Number of wait events. Configurable, must evenly divide max_wait_time.
    wait_events: int
    # Max wait time in milliseconds to be represented by a single token.
    max_wait_time: int
    # Number of velocity events. Should be 128 (or 100? need to check midi standard)
    velocity_events: int
    # Number of bins to quantize velocity into. Should evenly divide velocity_events.
    velocity_bins: int
    # Exponential scaling factor for velocity bin sizes. 1.0 = linear scaling.
    velocity_exp: float
    # Whether to sort tokens by instrument, note. This should improve data reducibility.
    do_token_sorting: bool
    # Whether tokens should be represented as combined instrument/note/velocity tokens, or separate tokens for each.
    unrolled_tokens: bool
    # If non-zero, notes held for this many seconds will be automatically released during str->midi decoding.
    decode_end_held_note_delay: float
    # If true, repeated notes will be automatically released before playing again during str->midi decoding.
    decode_fix_repeated_notes: bool
    # List of instrument names to use for binning. Must have at most 16 values.
    bin_instrument_names: List[str]
    # Indicates which bin name represents percussion instruments on MIDI channel 10.
    ch10_instrument_bin_name: str
    # Mapping from instrument name to bin name.
    program_name_to_bin_name: Dict[str, str]
    # Mapping from bin name to program name.
    bin_name_to_program_name: Dict[str, str]
    # Mapping from program number to instrument name.
    instrument_names: Dict[str, str]
    # Manual override for velocity bins. Each element is the max velocity value for that bin by index.
    velocity_bins_override: Optional[List[int]] = None

    def __post_init__(self):
        self.validate()

        self._instrument_names_str_to_int = {
            name: int(i) for i, name in self.instrument_names.items()
        }
        self._instrument_names_int_to_str = {
            int(i): name for i, name in self.instrument_names.items()
        }

        self._bin_str_to_int = {
            name: int(i) for i, name in enumerate(self.bin_instrument_names)
        }

        self._bin_int_to_instrument_int = [
            self._instrument_names_str_to_int[self.bin_name_to_program_name[name]]
            if name != self.ch10_instrument_bin_name
            else 0
            for name in self.bin_instrument_names
        ]
        self._instrument_int_to_bin_int = [
            self._bin_str_to_int[self.program_name_to_bin_name[instr]]
            if self.program_name_to_bin_name[instr] != ""
            else -1
            for instr in self.program_name_to_bin_name.keys()
        ]

        self._ch10_bin_int = (
            self._bin_str_to_int[self.ch10_instrument_bin_name]
            if self.ch10_instrument_bin_name
            else -1
        )

        self.short_instr_bin_names = []
        for instr in self.bin_instrument_names:
            i = min(1, len(instr))
            while instr[:i] in self.short_instr_bin_names:
                i += 1
            self.short_instr_bin_names.append(instr[:i])
        self._short_instrument_names_str_to_int = {
            name: int(i) for i, name in enumerate(self.short_instr_bin_names)
        }

        range_excluding_ch10 = [
            (i if i < 9 else i + 1) for i in range(len(self.bin_instrument_names))
        ]
        bins_excluding_ch10 = [
            n for n in self.bin_instrument_names if n != self.ch10_instrument_bin_name
        ]
        self.bin_channel_map = {
            bin: channel
            for channel, bin in zip(range_excluding_ch10, bins_excluding_ch10)
        }
        if self.ch10_instrument_bin_name:
            self.bin_channel_map[self.ch10_instrument_bin_name] = 9

    def validate(self):
        if self.max_wait_time % self.wait_events != 0:
            raise ValueError("max_wait_time must be exactly divisible by wait_events")
        if self.velocity_bins < 2:
            raise ValueError("velocity_bins must be at least 2")
        if len(self.bin_instrument_names) > 16:
            raise ValueError("bin_instruments must have at most 16 values")
        if self.velocity_bins_override:
            print("VocabConfig is using velocity_bins_override. Ignoring velocity_exp.")
            if len(self.velocity_bins_override) != self.velocity_bins:
                raise ValueError(
                    "velocity_bins_override must have same length as velocity_bins"
                )
        if (
            self.ch10_instrument_bin_name
            and self.ch10_instrument_bin_name not in self.bin_instrument_names
        ):
            raise ValueError("ch10_instrument_bin_name must be in bin_instruments")
        if self.velocity_exp <= 0:
            raise ValueError("velocity_exp must be greater than 0")

    @classmethod
    def from_json(cls, path: str):
        with open(path, "r") as f:
            config = json.load(f)
        return cls(**config)


class VocabUtils:
    def __init__(self, cfg: VocabConfig) -> None:
        self.cfg = cfg

    @lru_cache(maxsize=128)
    def format_wait_token(self, wait: int) -> str:
        return f"t{wait}"

    @lru_cache(maxsize=128)
    def format_note_token(
        self, instrument_bin: int, note: int, velocity_bin: int
    ) -> str:
        return f"{self.cfg.short_instr_bin_names[instrument_bin]}:{note:x}:{velocity_bin:x}"

    def format_unrolled_note(self, note: int) -> str:
        return f"n{note:x}"

    def format_unrolled_velocity(self, velocity_bin: int) -> str:
        return f"v{velocity_bin:x}"

    def format_unrolled_instrument_bin(self, instrument_bin: int) -> str:
        return f"i{self.cfg.short_instr_bin_names[instrument_bin]}"

    def velocity_to_bin(self, velocity: float) -> int:
        velocity = max(0, min(velocity, self.cfg.velocity_events - 1))
        if self.cfg.velocity_bins_override:
            for i, v in enumerate(self.cfg.velocity_bins_override):
                if velocity <= v:
                    return i
            return 0
        binsize = self.cfg.velocity_events / (self.cfg.velocity_bins - 1)
        if self.cfg.velocity_exp == 1.0:
            return ceil(velocity / binsize)
        else:
            return ceil(
                (
                    self.cfg.velocity_events
                    * (
                        (
                            self.cfg.velocity_exp
                            ** (velocity / self.cfg.velocity_events)
                            - 1.0
                        )
                        / (self.cfg.velocity_exp - 1.0)
                    )
                )
                / binsize
            )

    def bin_to_velocity(self, bin: int) -> int:
        if self.cfg.velocity_bins_override:
            return self.cfg.velocity_bins_override[bin]
        binsize = self.cfg.velocity_events / (self.cfg.velocity_bins - 1)
        if self.cfg.velocity_exp == 1.0:
            return max(0, ceil(bin * binsize - 1))
        else:
            return max(
                0,
                ceil(
                    self.cfg.velocity_events
                    * log(
                        ((self.cfg.velocity_exp - 1) * binsize * bin)
                        / self.cfg.velocity_events
                        + 1,
                        self.cfg.velocity_exp,
                    )
                    - 1
                ),
            )

    def delta_to_wait_ids(self, delta_ms: float) -> Iterator[int]:
        def roundi(f: float):
            return ceil(f - 0.5)

        max_wait_ms = self.cfg.max_wait_time
        div = max_wait_ms / self.cfg.wait_events

        # if delta_ms // max_wait_ms > 512:  # arbitrary limit to avoid excessive time_shifts
        #    raise ValueError("delta_time is too large")
        if delta_ms > max_wait_ms * 10:
            delta_ms = max_wait_ms * 10  # truncate time

        for _ in range(floor(delta_ms / max_wait_ms)):
            yield roundi(max_wait_ms / div)
        leftover_time_shift = roundi((delta_ms % max_wait_ms) / div)
        if leftover_time_shift > 0:
            yield leftover_time_shift

    def prog_data_to_token_data(
        self, program: int, channel: int, note: int, velocity: float
    ) -> Optional[Tuple[int, int, int]]:
        if channel == 9:
            if self.cfg._ch10_bin_int == -1:
                return None
            return self.cfg._ch10_bin_int, note, self.velocity_to_bin(velocity)

        instrument_bin = self.cfg._instrument_int_to_bin_int[program]
        if instrument_bin != -1:
            return instrument_bin, note, self.velocity_to_bin(velocity)
        return None

    def prog_data_list_to_token_data_list(
        self, data: List[Tuple[int, int, int, float]]
    ) -> Iterator[Tuple[int, int, int]]:
        for d in data:
            token_data = self.prog_data_to_token_data(*d)
            if token_data is not None:
                yield token_data

    def sort_token_data(
        self, data: List[Tuple[int, int, int]]
    ) -> List[Tuple[int, int, int]]:
        # ensure order is preserved for tokens with the same instrument, note
        data = [(i, n, v, x) for x, (i, n, v) in enumerate(data)]
        data.sort(key=lambda x: (x[0] != self.cfg._ch10_bin_int, x[0], x[1], x[3]))
        return [(i, n, v) for i, n, v, _ in data]

    def data_to_wait_tokens(self, delta_ms: float) -> List[str]:
        if delta_ms == 0.0:
            return []
        return [self.format_wait_token(i) for i in self.delta_to_wait_ids(delta_ms)]

    def wait_token_to_delta(self, token: str) -> float:
        return self.cfg.max_wait_time / self.cfg.wait_events * int(token[1:])

    def note_token_to_data(self, token: str) -> Tuple[int, int, int]:
        instr_str, note_str, velocity_str = token.strip().split(":")
        instr_bin = self.cfg._short_instrument_names_str_to_int[instr_str]
        note = int(note_str, base=16)
        velocity = self.bin_to_velocity(int(velocity_str, base=16))
        return instr_bin, note, velocity


@dataclass
class AugmentValues:
    instrument_bin_remap: Dict[int, int]
    velocity_mod_factor: float
    transpose_semitones: int
    time_stretch_factor: float

    @classmethod
    def default(cls) -> "AugmentValues":
        return cls(
            instrument_bin_remap={},
            velocity_mod_factor=1.0,
            transpose_semitones=0,
            time_stretch_factor=1.0,
        )


@dataclass
class AugmentConfig:
    # The number of times to augment each MIDI file. The dataset size will be multiplied by this number.
    augment_data_factor: int
    # A list of instrument names to randomly swap with each other.
    instrument_mixups: List[List[str]]
    # A list of percentages to change the note velocity by. 0.0 = no change. 0 is included by default.
    velocity_mod_pct: List[float]
    # A list of semitones to transpose by. 0 is included by default.
    transpose_semitones: List[int]
    # A list of percentages to stretch the tempo by. 0.0 = no stretch. 0 is included by default.
    time_stretch_pct: List[float]
    # Random seed to use for reproducibility.
    seed: int

    cfg: VocabConfig

    def __post_init__(self):
        self.validate()
        if len(self.velocity_mod_pct) == 0:
            self.velocity_mod_pct = [0.0]
        if len(self.transpose_semitones) == 0:
            self.transpose_semitones = [0]
        if len(self.time_stretch_pct) == 0:
            self.time_stretch_pct = [0.0]

        self._instrument_mixups_int = [
            [self.cfg._bin_str_to_int[i] for i in l if i in self.cfg._bin_str_to_int]
            for l in self.instrument_mixups
        ]
        self._instrument_mixups_int = [
            l for l in self._instrument_mixups_int if len(l) > 0
        ]  # remove empty lists
        self._instrument_pool_assignments = {}
        self._mixup_pools = []
        for pool_i, mixup_list in enumerate(self._instrument_mixups_int):
            pool = set()
            for i in mixup_list:
                pool.add(i)
                self._instrument_pool_assignments[i] = pool_i
            self._mixup_pools.append(pool)

    def validate(self):
        if self.augment_data_factor < 1:
            raise ValueError("augment_data_factor must be at least 1")
        used_instruments = set()
        for mixup_list in self.instrument_mixups:
            for n in mixup_list:
                if n in used_instruments:
                    raise ValueError(f"Duplicate instrument name: {n}")
                used_instruments.add(n)

    @classmethod
    def from_json(cls, path: str, cfg: VocabConfig):
        with open(path, "r") as f:
            config = json.load(f)
        config["cfg"] = cfg
        if "seed" not in config:
            config["seed"] = random.randint(0, 2**32 - 1)
        return cls(**config)

    def get_augment_values(self, filename: str) -> Iterator[AugmentValues]:
        # first yield default values
        yield AugmentValues.default()

        rng = random.Random(self.seed + hash(filename))
        for _ in range(int(self.augment_data_factor - 1)):
            # randomize order for each pool
            randomized_pools = [list(pool) for pool in self._mixup_pools]
            for pool in randomized_pools:
                rng.shuffle(pool)
            # distribute reassignments
            instrument_bin_remap = {}
            for i, pool in enumerate(randomized_pools):
                for j, instrument in enumerate(pool):
                    instrument_bin_remap[instrument] = randomized_pools[i - 1][j]
            yield AugmentValues(
                instrument_bin_remap=instrument_bin_remap,
                velocity_mod_factor=1.0 + rng.choice(self.velocity_mod_pct),
                transpose_semitones=rng.choice(self.transpose_semitones),
                time_stretch_factor=1.0 + rng.choice(self.time_stretch_pct),
            )


@dataclass
class FilterConfig:
    # Whether to filter out MIDI files with duplicate MD5 hashes.
    deduplicate_md5: bool
    # Minimum time delay between notes in a file before splitting into multiple documents.
    piece_split_delay: float
    # Minimum length of a piece in milliseconds.
    min_piece_length: float

    @classmethod
    def from_json(cls, path: str):
        with open(path, "r") as f:
            config = json.load(f)
        return cls(**config)


def mix_volume(velocity: int, volume: int, expression: int) -> float:
    return velocity * (volume / 127.0) * (expression / 127.0)


def convert_midi_to_str(
    cfg: VocabConfig,
    filter_cfg: FilterConfig,
    mid: mido.MidiFile,
    augment: AugmentValues = None,
) -> List[str]:
    utils = VocabUtils(cfg)
    if augment is None:
        augment = AugmentValues.default()

    # filter out unknown meta messages before merge (https://github.com/mido/mido/pull/286)
    for i in range(len(mid.tracks)):
        mid.tracks[i] = [msg for msg in mid.tracks[i] if msg.type != "unknown_meta"]

    if len(mid.tracks) > 1:
        mid.tracks = [mido.merge_tracks(mid.tracks)]

    delta_time_ms = 0.0
    tempo = 500000
    channel_program = {i: 0 for i in range(16)}
    channel_volume = {i: 127 for i in range(16)}
    channel_expression = {
        i: 127 for i in range(16)
    }  # unlikely to be useful. expression usually modifies an already played note.
    channel_notes = {i: {} for i in range(16)}
    channel_pedal_on = {i: False for i in range(16)}
    channel_pedal_events = {
        i: {} for i in range(16)
    }  # {channel: {(note, program) -> True}}
    started_flag = False

    output_list = []
    output = ["<start>"]
    output_length_ms = 0.0
    token_data_buffer: List[
        Tuple[int, int, int, float]
    ] = []  # need to sort notes between wait tokens

    def flush_token_data_buffer():
        nonlocal token_data_buffer, output, cfg, utils, augment
        token_data = [
            x for x in utils.prog_data_list_to_token_data_list(token_data_buffer)
        ]
        if augment.instrument_bin_remap or augment.transpose_semitones:
            # TODO put transpose in a real function
            raw_transpose = (
                lambda bin, n: n + augment.transpose_semitones
                if bin != cfg._ch10_bin_int
                else n
            )
            octave_shift_if_oob = (
                lambda n: n + 12 if n < 0 else n - 12 if n >= cfg.note_events else n
            )
            # TODO handle ranges beyond 12
            # octave_shift_if_oob = lambda n: 0 if n < 0 else (n - cfg.note_events) % 12 + cfg.note_events if n >= cfg.note_events else n
            transpose = lambda bin, n: octave_shift_if_oob(raw_transpose(bin, n))

            token_data = [
                (augment.instrument_bin_remap.get(i, i), transpose(i, n), v)
                for i, n, v in token_data
            ]
        if cfg.do_token_sorting:
            token_data = utils.sort_token_data(token_data)
        if cfg.unrolled_tokens:
            for t in token_data:
                output += [
                    utils.format_unrolled_instrument_bin(t[0]),
                    utils.format_unrolled_note(t[1]),
                    utils.format_unrolled_velocity(t[2]),
                ]
        else:
            output += [utils.format_note_token(*t) for t in token_data]
        token_data_buffer = []

    def consume_note_program_data(prog: int, chan: int, note: int, vel: float):
        nonlocal output, output_length_ms, started_flag, delta_time_ms, cfg, utils, token_data_buffer
        is_token_valid = (
            utils.prog_data_to_token_data(prog, chan, note, vel) is not None
        )
        if not is_token_valid:
            return

        if delta_time_ms > filter_cfg.piece_split_delay * 1000.0:
            # check if any notes are still held
            silent = True
            for channel in channel_notes.keys():
                if len(channel_notes[channel]) > 0:
                    silent = False
                    break
            if silent:
                flush_token_data_buffer()
                output.append("<end>")
                if output_length_ms > filter_cfg.min_piece_length * 1000.0:
                    output_list.append(" ".join(output))
                output = ["<start>"]
                output_length_ms = 0.0
                started_flag = False
        if started_flag:
            wait_tokens = utils.data_to_wait_tokens(delta_time_ms)
            if len(wait_tokens) > 0:
                flush_token_data_buffer()
                output_length_ms += delta_time_ms
                output += wait_tokens
        delta_time_ms = 0.0
        token_data_buffer.append((prog, chan, note, vel * augment.velocity_mod_factor))
        started_flag = True

    for msg in mid.tracks[0]:
        time_ms = mido.tick2second(msg.time, mid.ticks_per_beat, tempo) * 1000.0
        delta_time_ms += time_ms
        t = msg.type

        if msg.is_meta:
            if t == "set_tempo":
                tempo = msg.tempo * augment.time_stretch_factor
            continue

        def handle_note_off(ch, prog, n):
            if channel_pedal_on[ch]:
                channel_pedal_events[ch][(n, prog)] = True
            else:
                consume_note_program_data(prog, ch, n, 0)
                if n in channel_notes[ch]:
                    del channel_notes[ch][n]

        if t == "program_change":
            channel_program[msg.channel] = msg.program
        elif t == "note_on":
            if msg.velocity == 0:
                handle_note_off(msg.channel, channel_program[msg.channel], msg.note)
            else:
                if (msg.note, channel_program[msg.channel]) in channel_pedal_events[
                    msg.channel
                ]:
                    del channel_pedal_events[msg.channel][
                        (msg.note, channel_program[msg.channel])
                    ]
                consume_note_program_data(
                    channel_program[msg.channel],
                    msg.channel,
                    msg.note,
                    mix_volume(
                        msg.velocity,
                        channel_volume[msg.channel],
                        channel_expression[msg.channel],
                    ),
                )
                channel_notes[msg.channel][msg.note] = True
        elif t == "note_off":
            handle_note_off(msg.channel, channel_program[msg.channel], msg.note)
        elif t == "control_change":
            if msg.control == 7 or msg.control == 39:  # volume
                channel_volume[msg.channel] = msg.value
            elif msg.control == 11:  # expression
                channel_expression[msg.channel] = msg.value
            elif msg.control == 64:  # sustain pedal
                channel_pedal_on[msg.channel] = msg.value >= 64
                if not channel_pedal_on[msg.channel]:
                    for note, program in channel_pedal_events[msg.channel]:
                        handle_note_off(msg.channel, program, note)
                    channel_pedal_events[msg.channel] = {}
            elif msg.control == 123:  # all notes off
                for channel in channel_notes.keys():
                    for note in list(channel_notes[channel]).copy():
                        handle_note_off(channel, channel_program[channel], note)
        else:
            pass

    flush_token_data_buffer()
    output.append("<end>")
    if output_length_ms > filter_cfg.min_piece_length * 1000.0:
        output_list.append(" ".join(output))
    return output_list


def generate_program_change_messages(cfg: VocabConfig):
    for bin_name, channel in cfg.bin_channel_map.items():
        if channel == 9:
            continue
        program = cfg._instrument_names_str_to_int[
            cfg.bin_name_to_program_name[bin_name]
        ]
        yield mido.Message("program_change", program=program, time=0, channel=channel)
    yield mido.Message("program_change", program=0, time=0, channel=9)


@dataclass
class DecodeState:
    total_time: float  # milliseconds
    delta_accum: float  # milliseconds
    current_bin: int
    current_note: int
    active_notes: Dict[Tuple[int, int], float]  # { (channel, note): time started, ... }


def token_to_midi_message(
    utils: VocabUtils, token: str, state: DecodeState, end_token_pause: float = 3.0
) -> Iterator[Tuple[Optional[mido.Message], DecodeState]]:
    if state is None:
        state = DecodeState(
            total_time=0.0,
            delta_accum=0.0,
            current_bin=utils.cfg._short_instrument_names_str_to_int[
                utils.cfg.short_instr_bin_names[0]
            ],
            current_note=0,
            active_notes={},
        )
    token = token.strip()
    if not token:
        yield None, state
        return
    if token == "<end>":
        d = end_token_pause * 1000.0
        state.delta_accum += d
        state.total_time += d
        if utils.cfg.decode_end_held_note_delay != 0.0:
            # end held notes
            for (channel, note), start_time in list(state.active_notes.items()).copy():
                ticks = int(mido.second2tick(state.delta_accum / 1000.0, 480, 500000))
                state.delta_accum = 0.0
                del state.active_notes[(channel, note)]
                yield mido.Message(
                    "note_off", note=note, time=ticks, channel=channel
                ), state
        yield None, state
        return
    if token.startswith("<"):
        yield None, state
        return

    if utils.cfg.unrolled_tokens:
        if token[0] == "t":
            d = utils.wait_token_to_delta(token)
            state.delta_accum += d
            state.total_time += d
        elif token[0] == "n":
            state.current_note = int(token[1:], base=16)
        elif token[0] == "i":
            state.current_bin = utils.cfg._short_instrument_names_str_to_int[token[1:]]
        elif token[0] == "v":
            current_velocity = utils.bin_to_velocity(int(token[1:], base=16))
            channel = utils.cfg.bin_channel_map[
                utils.cfg.bin_instrument_names[state.current_bin]
            ]
            ticks = int(mido.second2tick(state.delta_accum / 1000.0, 480, 500000))
            state.delta_accum = 0.0
            if current_velocity > 0:
                yield mido.Message(
                    "note_on",
                    note=state.current_note,
                    velocity=current_velocity,
                    time=ticks,
                    channel=channel,
                ), state
            else:
                yield mido.Message(
                    "note_off",
                    note=state.current_note,
                    velocity=0,
                    time=ticks,
                    channel=channel,
                ), state
    else:
        if token[0] == "t" and token[1].isdigit():  # wait token
            d = utils.wait_token_to_delta(token)
            state.delta_accum += d
            state.total_time += d
            if utils.cfg.decode_end_held_note_delay != 0.0:
                # remove notes that have been held for too long
                for (channel, note), start_time in list(
                    state.active_notes.items()
                ).copy():
                    if (
                        state.total_time - start_time
                        > utils.cfg.decode_end_held_note_delay * 1000.0
                    ):
                        ticks = int(
                            mido.second2tick(state.delta_accum / 1000.0, 480, 500000)
                        )
                        state.delta_accum = 0.0
                        del state.active_notes[(channel, note)]
                        yield mido.Message(
                            "note_off", note=note, time=ticks, channel=channel
                        ), state
                        return
        else:  # note token
            bin, note, velocity = utils.note_token_to_data(token)
            channel = utils.cfg.bin_channel_map[utils.cfg.bin_instrument_names[bin]]
            ticks = int(mido.second2tick(state.delta_accum / 1000.0, 480, 500000))
            state.delta_accum = 0.0
            if velocity > 0:
                if utils.cfg.decode_fix_repeated_notes:
                    if (channel, note) in state.active_notes:
                        del state.active_notes[(channel, note)]
                        yield mido.Message(
                            "note_off", note=note, time=ticks, channel=channel
                        ), state
                        ticks = 0
                state.active_notes[(channel, note)] = state.total_time
                yield mido.Message(
                    "note_on", note=note, velocity=velocity, time=ticks, channel=channel
                ), state
                return
            else:
                if (channel, note) in state.active_notes:
                    del state.active_notes[(channel, note)]
                yield mido.Message(
                    "note_off", note=note, time=ticks, channel=channel
                ), state
                return
    yield None, state


def str_to_midi_messages(utils: VocabUtils, data: str) -> Iterator[mido.Message]:
    state = None
    for token in data.split(" "):
        for msg, new_state in token_to_midi_message(utils, token, state):
            state = new_state
            if msg is not None:
                yield msg


def convert_str_to_midi(
    cfg: VocabConfig, data: str, meta_text: str = "Generated by MIDI-LLM-tokenizer"
) -> mido.MidiFile:
    utils = VocabUtils(cfg)
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    tempo = 500000
    if meta_text:
        track.append(mido.MetaMessage("text", text=meta_text, time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    for msg in generate_program_change_messages(cfg):
        track.append(msg)

    # data = data.replace("<start>", "").replace("<end>", "").replace("<pad>", "").strip()
    for msg in str_to_midi_messages(utils, data):
        track.append(msg)

    track.append(mido.MetaMessage("end_of_track", time=0))

    return mid
