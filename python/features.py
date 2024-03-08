from typing import List, Tuple, Union
import partitura as pt
import numpy as np
from itertools import combinations


def get_pc_one_hot(part, note_array):
    one_hot = np.zeros((len(note_array), 12))
    idx = (np.arange(len(note_array)),np.remainder(note_array["pitch"], 12))
    one_hot[idx] = 1
    return one_hot, ["pc_{:02d}".format(i) for i in range(12)]


def get_full_pitch_one_hot(part, note_array, piano_range = True):
    one_hot = np.zeros((len(note_array), 127))
    idx = (np.arange(len(note_array)),note_array["pitch"])
    one_hot[idx] = 1
    if piano_range:
        one_hot = one_hot[:, 21:109]
    return one_hot, ["pc_{:02d}".format(i) for i in range(one_hot.shape[1])]


def get_octave_one_hot(part, note_array):
    one_hot = np.zeros((len(note_array), 10))
    idx = (np.arange(len(note_array)), np.floor_divide(note_array["pitch"], 12))
    one_hot[idx] = 1
    return one_hot, ["octave_{:02d}".format(i) for i in range(10)]


def chord_to_intervalVector(midi_pitches, return_pc_class=False):
    '''Given a chord it calculates the Interval Vector.


    Parameters
    ----------
    midi_pitches : list(int)
        The midi_pitches, is a list of integers < 128.

    Returns
    -------
    intervalVector : list(int)
        The interval Vector is a list of six integer values.
    '''
    intervalVector = [0, 0, 0, 0, 0, 0]
    PC = set([mp%12 for mp in midi_pitches])
    for p1, p2 in combinations(PC, 2):
        interval = int(abs(p1 - p2))
        if interval <= 6:
            index = interval
        else:
            index = 12 - interval
        if index != 0:
            index = index-1
            intervalVector[index] += 1
    if return_pc_class:
        return intervalVector, list(PC)
    else:
        return intervalVector


def get_cad_features(part, note_array, include_pitch_spelling=True):
    """
    Create cadence relevant features on the note level.

    Parameters
    ----------
    part : partitura.score.Part
        In this function a dummy variable. It can be given empty.
    note_array : numpy structured array
        A part note array. Attention part must contain time signature information.
    include_pitch_spelling : bool
        Whether to include pitch spelling information in the features.

    Returns
    -------
    feat_array : numpy structured array
        A structured array of features. Each line corresponds to a note in the note array.
    feature_fn : list
        A list of the feature names.
    """

    features = list()
    bass_voice = note_array["voice"].max() if note_array["voice" == note_array["voice"].max()]["pitch"].mean() < note_array["voice" == note_array["voice"].min()]["pitch"].mean() else note_array["voice"].min()
    high_voice = note_array["voice"].min() if note_array["voice" == note_array["voice"].min()]["pitch"].mean() > \
                                              note_array["voice" == note_array["voice"].max()]["pitch"].mean() else note_array["voice"].max()
    for i, n in enumerate(note_array):
        d = {}
        n_onset = note_array[note_array["onset_div"] == n["onset_div"]]
        n_dur = note_array[np.where((note_array["onset_div"] < n["onset_div"]) & (note_array["onset_div"] + note_array["duration_div"] > n["onset_div"]))]
        chord_pitch = np.hstack((n_onset["pitch"], n_dur["pitch"]))
        int_vec, pc_class = chord_to_intervalVector(chord_pitch.tolist(), return_pc_class=True)
        pc_class_recentered = sorted(list(map(lambda x: x - min(pc_class), pc_class)))
        maj_int_vecs = [[0, 0, 1, 1, 1, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0]]
        prev_4beats = note_array[np.where((note_array["onset_beat"] < n["onset_beat"]) & (note_array["onset_beat"] > n["onset_beat"] - 4))][
                          "pitch"] % 12
        prev_8beats = note_array[np.where((note_array["onset_beat"] < n["onset_beat"]) & (note_array["onset_beat"] > n["onset_beat"] - 8))][
                          "pitch"] % 12
        maj_pcs = [[0, 4, 7], [0, 5, 9], [0, 3, 8], [0, 4], [0, 8], [0, 7], [0, 5]]
        scale = [2, 3, 5, 7, 8, 11] if (n["pitch"] + 3) in chord_pitch % 12 else [2, 4, 5, 7, 9, 11]
        v7 = [[0, 1, 2, 1, 1, 1], [0, 1, 0, 1, 0, 1], [0, 1, 0, 0, 0, 0]]
        next_voice_notes = note_array[np.where((note_array["voice"] == n["voice"]) & (note_array["onset_beat"] > n["onset_beat"]))]
        prev_voice_notes = note_array[np.where((note_array["voice"] == n["voice"]) & (note_array["onset_beat"] < n["onset_beat"]))]
        prev_voice_pitch = prev_voice_notes[prev_voice_notes["onset_beat"] == prev_voice_notes["onset_beat"].max()]["pitch"] if prev_voice_notes.size else 0
        # start Z features
        d["perfect_triad"] = int_vec in maj_int_vecs
        d["perfect_major_triad"] = d["perfect_triad"] and pc_class_recentered in maj_pcs
        d["is_sus4"] = int_vec == [0, 1, 0, 0, 2, 0] or pc_class_recentered == [0, 5]
        d["in_perfect_triad_or_sus4"] = d["perfect_triad"] or d["is_sus4"]
        d["highest_is_3"] = (chord_pitch.max() - chord_pitch.min()) % 12 in [3, 4]
        d["highest_is_1"] = (chord_pitch.max() - chord_pitch.min()) % 12 == 0 and chord_pitch.max() != chord_pitch.min()

        d["bass_compatible_with_I"] = (n["pitch"] + 5) % 12 in prev_4beats and (n["pitch"] + 11) % 12 in prev_4beats if prev_4beats.size else False
        d["bass_compatible_with_I_scale"] = all([(n["pitch"] + ni) % 12 in prev_8beats for ni in scale]) if prev_8beats.size else False
        d["one_comes_from_7"] = 11 in (prev_voice_pitch - chord_pitch.min())%12 and (
                n["pitch"] - chord_pitch.min())%12 == 0 if prev_voice_notes.size and len(chord_pitch)>1 else False
        d["one_comes_from_1"] = 0 in (prev_voice_pitch - chord_pitch.min())%12 and (
                    n["pitch"] - chord_pitch.min())%12 == 0 if prev_voice_notes.size and len(chord_pitch)>1 else False
        d["one_comes_from_2"] = 2 in (prev_voice_pitch - chord_pitch.min()) % 12 and (
                n["pitch"] - chord_pitch.min())%12 == 0 if prev_voice_notes.size and len(chord_pitch)>1 else False
        d["three_comes_from_4"] = 5 in (prev_voice_pitch - chord_pitch.min()) % 12 and (
                n["pitch"] - chord_pitch.min())%12 in [3, 4] if prev_voice_notes.size else False
        d["five_comes_from_5"] = 7 in (prev_voice_pitch - chord_pitch.min()) % 12 and (
                n["pitch"] - chord_pitch.min()) % 12 == 7 if prev_voice_notes.size else False

        # Make R features
        d["strong_beat"] = (n["ts_beats"] == 4 and n["onset_beat"] % 2 == 0) or (n["onset_beat"] % n['ts_beats'] == 0) # to debug
        d["sustained_note"] = n_dur.size > 0
        if next_voice_notes.size:
            d["rest_highest"] = n["voice"] == high_voice and next_voice_notes["onset_beat"].min() > n["onset_beat"] + n["duration_beat"]
            d["rest_lowest"] = n["voice"] == bass_voice and next_voice_notes["onset_beat"].min() > n["onset_beat"] + n["duration_beat"]
            d["rest_middle"] = n["voice"] != high_voice and n["voice"] != bass_voice and next_voice_notes["onset_beat"].min() > n[
                "onset_beat"] + n["duration_beat"]
            d["voice_ends"] = False
        else:
            d["rest_highest"] = False
            d["rest_lowest"] = False
            d["rest_middle"] = False
            d["voice_ends"] = True
        d["is_downbeat"] = n["is_downbeat"]
        # start Y features
        d["v7"] = int_vec in v7
        d["v7-3"] = int_vec in v7 and 4 in pc_class_recentered
        d["has_7"] = 10 in pc_class_recentered
        d["has_9"] = 1 in pc_class_recentered or 2 in pc_class_recentered
        d["bass_voice"] = n["voice"] == bass_voice
        if prev_voice_notes.size:
            x = prev_voice_notes[prev_voice_notes["onset_beat"] == prev_voice_notes["onset_beat"].max()]["pitch"]
            d["bass_moves_chromatic"] = n["voice"] == bass_voice and (1 in x - n["pitch"] or -1 in x-n["pitch"])
            d["bass_moves_octave"] = n["voice"] == bass_voice and (12 in x - n["pitch"] or -12 in x - n["pitch"])
            d["bass_compatible_v-i"] = n["voice"] == bass_voice and (7 in x - n["pitch"] or -5 in x - n["pitch"])
            d["bass_compatible_i-v"] = n["voice"] == bass_voice and (-7 in x - n["pitch"] or 5 in x - n["pitch"])
        # X features
            d["bass_moves_2M"] = n["voice"] == bass_voice and (2 in x - n["pitch"] or -2 in x - n["pitch"])
        else:
            d["bass_moves_chromatic"] = d["bass_moves_octave"] = d["bass_compatible_v-i"] = d["bass_compatible_i-v"] = d["bass_moves_2M"] = False
        features.append(tuple(d.values()))

    feat_array = np.array(features)
    feature_fn = list(d.keys())

    return feat_array, feature_fn


def get_voice_separation_features(part) -> Tuple[np.ndarray, List]:
    """
    Returns features Voice Detection features.

    Parameters
    ----------
    part: Part, PartGroup or PerformedPart

    Returns
    -------
    out : np.ndarray
    feature_fn : List
    """
    if isinstance(part, pt.performance.PerformedPart):
        perf_array = part.note_array()
        x = perf_array[["onset_sec", "duration_sec"]].astype([("onset_beat", "f4"), ("duration_beat", "f4")])
        note_array = np.lib.recfunctions.merge_arrays((perf_array, x))
    elif isinstance(part, np.ndarray):
        note_array = part
        part = None
    else:
        note_array = part.note_array(include_time_signature=True)

    # octave_oh, octave_names = get_octave_one_hot(part, note_array)
    # pc_oh, pc_names = get_pc_one_hot(part, note_array)
    # onset_feature = np.expand_dims(np.remainder(note_array["onset_beat"], note_array["ts_beats"]) / note_array["ts_beats"], 1)
    # on_feats, _ = pt.musicanalysis.note_features.onset_feature(note_array, part)
    # duration_feature = np.expand_dims(np.remainder(note_array["duration_beat"], note_array["ts_beats"]) / note_array["ts_beats"], 1)
    # # new attempt! To delete in case
    # # duration_feature = np.expand_dims(1- (1/(1+np.exp(-3*(note_array["duration_beat"]/note_array["ts_beats"])))-0.5)*2, 1)
    # pitch_norm = np.expand_dims(note_array["pitch"] / 127., 1)
    # on_names = ["barnorm_onset", "piecenorm_onset"]
    # dur_names = ["barnorm_duration"]
    # pitch_names = ["pitchnorm"]
    # names = on_names + dur_names + pitch_names + pc_names + octave_names
    # out = np.hstack((onset_feature, np.expand_dims(on_feats[:, 1], 1), duration_feature, pitch_norm, pc_oh, octave_oh))

    # octave_oh, octave_names = get_octave_one_hot(part, note_array)
    # pitch_oh, pitch_names = get_full_pitch_one_hot(part, note_array)
    # onset_feature = np.expand_dims(np.remainder(note_array["onset_beat"], note_array["ts_beats"]) / note_array["ts_beats"], 1)
    # on_feats, _ = pt.musicanalysis.note_features.onset_feature(note_array, part)
    octave_oh, octave_names = get_octave_one_hot(part, note_array)
    pc_oh, pc_names = get_pc_one_hot(part, note_array)
    # duration_feature = np.expand_dims(1- (1/(1+np.exp(-3*(note_array["duration_beat"]/note_array["ts_beats"])))-0.5)*2, 1)
    duration_feature = np.expand_dims(1 - np.tanh(note_array["duration_beat"]/note_array["ts_beats"]), 1)
    dur_names = ["bar_exp_duration"]
    # on_names = ["barnorm_onset", "piecenorm_onset"]
    names = dur_names + pc_names + octave_names
    out = np.hstack((duration_feature, pc_oh, octave_oh))
    return out, names


def cadence_features(part: Union[Union[pt.score.Part, pt.score.PartGroup], pt.performance.PerformedPart]) -> Tuple[np.ndarray, List]:
    voice_features, voice_names = get_voice_separation_features(part)
    if isinstance(part, np.ndarray):
        note_array = part
    else:
        note_array = part.note_array(
            include_time_signature=True,
            include_staff=True,
            include_grace_notes=True,
            include_metrical_position=True,
            include_pitch_spelling=True
            )
    cad_features, cad_names = get_cad_features(None, note_array, include_pitch_spelling=False)
    features = np.hstack((voice_features, cad_features))
    fnames = voice_names + cad_names
    return features, fnames